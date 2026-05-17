"""Phase 4A quote-history persistence + A.11 feature computation.

Per tick, we:

1. Snapshot every loaded candidate's quote into ``quote_history``.
2. Read the last N (default 8) snapshots back per market and compute
   the features listed in AGENTS.md A.11.1.
3. Translate features into three optional channels (A.11.2):
   - ``qh_confidence_modifier`` -- multiplies the A.5 score.
   - ``deference_extra_shrinkage`` -- feeds ``optional_extra_shrinkage``
     in :func:`kalibre.forecast.blend.blend_with_market_prior`.
   - ``exit_pressure`` -- adjusts hold/exit decisions.

All three channels are gated by layer weights in
:class:`kalibre.layers.QuoteHistoryLayer`. Default weights = 0.0 so
Phase 3B behavior is preserved unless an operator explicitly enables
them.

Persistence is best-effort: failure to compute features for a single
market must never block forecasting.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable

from kalibre.forecast.blend import (
    blend_with_market_prior,
    compute_market_prior,
)
from kalibre.layers import QuoteHistoryLayer
from kalibre.tick_context import MarketView, TickContext


logger = logging.getLogger("kalibre.quote_history")


DEFAULT_HISTORY_WINDOW = 8
SMART_SPREAD_MAX = 0.02
SMART_VOLUME_MIN = 5000.0
SMART_STABILITY_MIN = 0.7


# --- raw snapshot ----------------------------------------------------------


@dataclass(frozen=True)
class QuoteSnapshot:
    market_id: str
    tick_ts: str
    mid: float | None
    spread: float | None
    best_bid: float | None
    best_ask: float | None
    volume_24h: float | None
    quote_ts: str | None


def _ts_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def build_quote_snapshot_rows(
    *,
    ctx: TickContext,
    markets: Iterable[MarketView],
) -> list[dict[str, Any]]:
    """One quote_history row per loaded candidate."""
    rows: list[dict[str, Any]] = []
    created = datetime.now(tz=UTC).isoformat()
    for market in markets:
        q = market.quote
        rows.append({
            "market_id": market.market_id,
            "tick_ts": ctx.tick_id,
            "experiment_id": ctx.experiment_id,
            "best_bid": q.best_bid,
            "best_ask": q.best_ask,
            "mid": q.mid,
            "spread": q.spread,
            "volume_24h": q.volume_24h,
            "quote_ts": _ts_iso(q.ts),
            "quote_age_sec": q.quote_age_sec(ctx.now),
            "source": market.source,
            "topic": market.topic,
            "created_at": created,
        })
    return rows


# --- A.11 features ---------------------------------------------------------


@dataclass(frozen=True)
class QuoteFeatures:
    market_id: str
    tick_ts: str
    mid_return_1_tick: float
    mid_return_4_tick: float
    mid_return_8_tick: float
    spread_current: float | None
    spread_change_4_tick: float
    quote_stability: float
    volatility_8_tick: float
    time_since_seen: float | None
    qh_warmup: bool
    market_is_smart: bool
    qh_confidence_modifier: float
    deference_extra_shrinkage: float
    exit_pressure: float
    history_len: int = 0

    def to_audit_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["qh_warmup"] = bool(out["qh_warmup"])
        out["market_is_smart"] = bool(out["market_is_smart"])
        return out

    def to_row(self, *, experiment_id: str, created_at: str | None = None) -> dict[str, Any]:
        created = created_at or datetime.now(tz=UTC).isoformat()
        return {
            "market_id": self.market_id,
            "tick_ts": self.tick_ts,
            "experiment_id": experiment_id,
            "mid_return_1_tick": self.mid_return_1_tick,
            "mid_return_4_tick": self.mid_return_4_tick,
            "mid_return_8_tick": self.mid_return_8_tick,
            "spread_current": self.spread_current,
            "spread_change_4_tick": self.spread_change_4_tick,
            "quote_stability": self.quote_stability,
            "volatility_8_tick": self.volatility_8_tick,
            "time_since_seen": self.time_since_seen,
            "qh_warmup": int(self.qh_warmup),
            "market_is_smart": int(self.market_is_smart),
            "qh_confidence_modifier": self.qh_confidence_modifier,
            "deference_extra_shrinkage": self.deference_extra_shrinkage,
            "exit_pressure": self.exit_pressure,
            "features_json": json.dumps(self.to_audit_dict(), default=str),
            "created_at": created,
        }


def _safe_return(now_mid: float | None, then_mid: float | None) -> float:
    if now_mid is None or then_mid is None:
        return 0.0
    if then_mid <= 0:
        return 0.0
    return (float(now_mid) - float(then_mid)) / float(then_mid)


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


def fetch_history(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    tick_ts: str,
    experiment_id: str,
    limit: int = DEFAULT_HISTORY_WINDOW,
) -> list[dict[str, Any]]:
    """Most-recent ``limit`` snapshots for ``market_id`` strictly before ``tick_ts``."""
    cur = conn.execute(
        "SELECT tick_ts, mid, spread, best_bid, best_ask, volume_24h, quote_ts "
        "FROM quote_history "
        "WHERE market_id=? AND experiment_id=? AND tick_ts < ? "
        "ORDER BY tick_ts DESC LIMIT ?",
        (market_id, experiment_id, tick_ts, int(limit)),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def compute_features(
    *,
    current: dict[str, Any],
    history: list[dict[str, Any]],
    layer: QuoteHistoryLayer | None = None,
    market_topic: str | None = None,
) -> QuoteFeatures:
    """Compute A.11 features from the current snapshot + historical rows.

    ``history`` is most-recent-first (length 0..N), excluding the current
    tick. The function honors the A.11 warmup rule: when fewer than 8
    historical ticks exist, longer-horizon returns clamp to 0 and
    ``qh_warmup`` is True.
    """
    layer = layer or QuoteHistoryLayer()
    now_mid = current.get("mid")
    now_spread = current.get("spread")
    n_history = len(history)
    qh_warmup = n_history < DEFAULT_HISTORY_WINDOW

    # History is newest-first; convenience indexed access:
    def _hist(i: int) -> dict[str, Any] | None:
        return history[i] if i < n_history else None

    one_back = _hist(0)
    four_back = _hist(3)
    eight_back = _hist(7)

    mid_return_1 = _safe_return(now_mid, one_back["mid"] if one_back else None)
    # AGENTS.md A.11.1 warmup rule: fewer than 8 historical ticks -> the
    # 4- and 8-tick returns must be 0, regardless of whether four_back
    # exists. Stale partial-window returns are misleading early.
    if qh_warmup:
        mid_return_4 = 0.0
        mid_return_8 = 0.0
    else:
        mid_return_4 = _safe_return(now_mid, four_back["mid"] if four_back else None)
        mid_return_8 = _safe_return(now_mid, eight_back["mid"] if eight_back else None)

    spread_change_4 = 0.0
    if not qh_warmup and four_back is not None and four_back.get("spread") is not None and now_spread is not None:
        spread_change_4 = float(now_spread) - float(four_back["spread"])

    # quote_stability = 1 / (1 + stdev(mid_{t-7..t})). We require <=8 mid
    # observations (current + up to 7 history). If we don't have at least
    # 2 observations we report neutral stability.
    midline = [float(current.get("mid"))] if current.get("mid") is not None else []
    for row in history[:7]:
        if row.get("mid") is not None:
            midline.append(float(row["mid"]))
    quote_stability = 1.0 / (1.0 + _stdev(midline)) if midline else 1.0

    # volatility_8_tick = stdev(mid_return_1_tick over last 8 ticks).
    one_tick_returns: list[float] = []
    chain = ([{"mid": current.get("mid")}] + history)[:9]
    for i in range(len(chain) - 1):
        one_tick_returns.append(_safe_return(chain[i].get("mid"), chain[i + 1].get("mid")))
    volatility_8 = _stdev(one_tick_returns)

    quote_ts = current.get("quote_ts")
    last_seen_age: float | None = None
    if isinstance(quote_ts, str):
        try:
            ts_dt = datetime.fromisoformat(quote_ts)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=UTC)
            # Approximate "now" with the current tick_ts; persisted in audit anyway.
            now_dt = datetime.fromisoformat(current["tick_ts"]) if isinstance(current.get("tick_ts"), str) else datetime.now(tz=UTC)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=UTC)
            last_seen_age = max(0.0, (now_dt - ts_dt).total_seconds())
        except (ValueError, KeyError):
            last_seen_age = None

    # market_is_smart per AGENTS.md A.11.3.
    volume = current.get("volume_24h")
    market_is_smart = bool(
        now_spread is not None and float(now_spread) <= SMART_SPREAD_MAX
        and quote_stability > SMART_STABILITY_MIN
        and volume is not None and float(volume) >= SMART_VOLUME_MIN
    )

    # --- A.11.2 channel formulas ------------------------------------------

    # Channel 1: confidence modifier (multiplies A.5 score). Clipped to
    # [0.70, 1.20] per spec. The layer weight in [0,1] interpolates from
    # the no-op value 1.0 toward the formula value.
    raw_conf = 1.0
    raw_conf += 0.10 * (quote_stability - 0.5)
    raw_conf -= 0.15 * (volatility_8 / 0.05)
    if spread_change_4 > 0.02:
        raw_conf -= 0.10
    raw_conf = min(1.20, max(0.70, raw_conf))
    if qh_warmup:
        raw_conf *= 0.85
    weight = max(0.0, min(1.0, layer.confidence_modifier_weight))
    qh_confidence_modifier = 1.0 + weight * (raw_conf - 1.0)

    # Channel 2: deference extra shrinkage (passed to the blender's
    # optional_extra_shrinkage). Smart markets get a +0.15 boost ramped
    # by the layer weight in [0, 1]. The blender additionally clips the
    # value into [0, 0.5] so this is safe even if weights ramp beyond 1.
    deference_extra_shrinkage = 0.0
    if market_is_smart:
        deference_extra_shrinkage = max(0.0, min(1.0, layer.deference_signal_weight)) * 0.15

    # Channel 3: exit_pressure for A.10 hold/exit (sign convention from
    # AGENTS.md A.11.2).
    direction_long = 1.0  # caller maps to position direction; we report magnitude only.
    exit_pressure_raw = 0.0
    if mid_return_4 > 0.0 and spread_change_4 < 0:
        exit_pressure_raw -= 0.05
    if mid_return_4 < 0.0 and spread_change_4 > 0.02:
        exit_pressure_raw += 0.10
    exit_pressure = max(0.0, min(1.0, layer.exit_signal_weight)) * exit_pressure_raw

    return QuoteFeatures(
        market_id=current["market_id"],
        tick_ts=current["tick_ts"],
        mid_return_1_tick=mid_return_1,
        mid_return_4_tick=mid_return_4,
        mid_return_8_tick=mid_return_8,
        spread_current=now_spread,
        spread_change_4_tick=spread_change_4,
        quote_stability=quote_stability,
        volatility_8_tick=volatility_8,
        time_since_seen=last_seen_age,
        qh_warmup=qh_warmup,
        market_is_smart=market_is_smart,
        qh_confidence_modifier=qh_confidence_modifier,
        deference_extra_shrinkage=deference_extra_shrinkage,
        exit_pressure=exit_pressure,
        history_len=n_history,
    )


# --- per-tick orchestration -----------------------------------------------


@dataclass(frozen=True)
class QuoteHistoryPassResult:
    snapshot_rows: list[dict[str, Any]] = field(default_factory=list)
    feature_rows: list[dict[str, Any]] = field(default_factory=list)
    features_by_market: dict[str, QuoteFeatures] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "snapshots": len(self.snapshot_rows),
            "features": len(self.feature_rows),
            "warmup_count": sum(1 for f in self.features_by_market.values() if f.qh_warmup),
            "smart_count": sum(1 for f in self.features_by_market.values() if f.market_is_smart),
        }


def run_quote_history_pass(
    *,
    conn: sqlite3.Connection,
    ctx: TickContext,
    markets: Iterable[MarketView],
    layer: QuoteHistoryLayer | None = None,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> QuoteHistoryPassResult:
    """Compute features for every loaded market.

    The caller is responsible for persisting the result via the state
    helpers (``record_quote_history`` + ``record_quote_features``).
    Persistence is split out so the runner can keep its database access
    in :mod:`kalibre.loop`.
    """
    layer = layer or QuoteHistoryLayer()
    markets = list(markets)
    snapshot_rows = build_quote_snapshot_rows(ctx=ctx, markets=markets)
    feature_rows: list[dict[str, Any]] = []
    features_by_market: dict[str, QuoteFeatures] = {}
    if not markets:
        return QuoteHistoryPassResult(
            snapshot_rows=snapshot_rows,
            feature_rows=feature_rows,
            features_by_market=features_by_market,
        )
    by_id = {r["market_id"]: r for r in snapshot_rows}
    for market in markets:
        current = by_id.get(market.market_id)
        if current is None:
            continue
        try:
            history = fetch_history(
                conn,
                market_id=market.market_id,
                tick_ts=ctx.tick_id,
                experiment_id=ctx.experiment_id,
                limit=history_window,
            )
        except sqlite3.OperationalError:
            history = []
        features = compute_features(
            current=current,
            history=history,
            layer=layer,
            market_topic=market.topic,
        )
        features_by_market[market.market_id] = features
        feature_rows.append(features.to_row(experiment_id=ctx.experiment_id))
    return QuoteHistoryPassResult(
        snapshot_rows=snapshot_rows,
        feature_rows=feature_rows,
        features_by_market=features_by_market,
    )


# --- Phase 4A repair: shadow ablation variants ----------------------------


SHADOW_FEATURES = "quote_history_features"
SHADOW_DEFERENCE = "quote_history_deference"
SHADOW_CONFIDENCE = "quote_history_confidence"
SHADOW_DEFERENCE_BOOST = 0.15  # AGENTS.md A.11.2 channel 2 max


def _shadow_envelope(
    *,
    ctx: TickContext,
    market_id: str,
    variant_name: str,
    edge_source: str,
    payload: dict[str, Any],
    p_mean: float | None = None,
    sigma_p: float | None = None,
    side: str | None = None,
    score: float | None = None,
    decision: str = "shadow",
    reject_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "variant_name": variant_name,
        "tick_ts": ctx.tick_id,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": market_id,
        "edge_source": edge_source,
        "p_mean": p_mean,
        "sigma_p": sigma_p,
        "side": side,
        "score": score,
        "decision": decision,
        "reject_reason": reject_reason,
        "audit_json": json.dumps(payload, default=str),
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


def build_features_shadow_rows(
    *,
    ctx: TickContext,
    features_by_market: dict[str, QuoteFeatures],
) -> list[dict[str, Any]]:
    """One ``quote_history_features`` shadow row per candidate with features.

    The audit_json carries the full feature dict so analysts can rebuild
    every channel from a single shadow row.
    """
    rows: list[dict[str, Any]] = []
    for market_id, features in features_by_market.items():
        payload = features.to_audit_dict()
        payload["variant"] = SHADOW_FEATURES
        rows.append(_shadow_envelope(
            ctx=ctx,
            market_id=market_id,
            variant_name=SHADOW_FEATURES,
            edge_source=SHADOW_FEATURES,
            payload=payload,
        ))
    return rows


def _market_horizon_hours(market: MarketView, ctx: TickContext) -> float | None:
    return market.hours_to_resolution(ctx.now)


def _confidence_modifier_raw(features: QuoteFeatures) -> float:
    """Compute A.11.2 channel-1 modifier value (clipped) at full weight."""
    raw = 1.0
    raw += 0.10 * (features.quote_stability - 0.5)
    raw -= 0.15 * (features.volatility_8_tick / 0.05)
    if features.spread_change_4_tick > 0.02:
        raw -= 0.10
    raw = min(1.20, max(0.70, raw))
    if features.qh_warmup:
        raw *= 0.85
    return raw


def build_deference_shadow_rows(
    *,
    ctx: TickContext,
    markets_by_id: dict[str, MarketView],
    features_by_market: dict[str, QuoteFeatures],
    forecast_audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One ``quote_history_deference`` shadow per forecasted market.

    Audit_json records the counterfactual blend that *would* have run if
    ``layers.quote_history.deference_signal_weight`` had been 1.0. The
    primary blend is unchanged.
    """
    rows: list[dict[str, Any]] = []
    for forecast_row in forecast_audit_rows:
        if forecast_row.get("error"):
            continue
        if forecast_row.get("decision") not in ("ok", "cached"):
            continue
        market_id = forecast_row.get("market_id")
        market = markets_by_id.get(market_id)
        if market is None:
            continue
        features = features_by_market.get(market_id)
        p_cal = forecast_row.get("p_cal")
        sigma_model = forecast_row.get("sigma_p")
        if p_cal is None or sigma_model is None:
            continue
        q = market.quote
        p_market = compute_market_prior(best_bid=q.best_bid, best_ask=q.best_ask)
        # Counterfactual: full deference weight on a smart market.
        extra_at_full_weight = SHADOW_DEFERENCE_BOOST if (features and features.market_is_smart) else 0.0
        counterfactual = blend_with_market_prior(
            p_model_cal=float(p_cal),
            sigma_model=float(sigma_model),
            p_market=p_market,
            spread=q.spread,
            volume_24h=q.volume_24h,
            time_to_resolve_hours=_market_horizon_hours(market, ctx),
            optional_extra_shrinkage=extra_at_full_weight,
        )
        baseline_p_blend = forecast_row.get("p_blend")
        delta = (
            counterfactual.p_blend - float(baseline_p_blend)
            if baseline_p_blend is not None else None
        )
        payload = {
            "variant": SHADOW_DEFERENCE,
            "deference_signal_weight": 1.0,
            "deference_extra_shrinkage": extra_at_full_weight,
            "market_is_smart": bool(features.market_is_smart) if features else None,
            "baseline_p_blend": baseline_p_blend,
            "baseline_blend_reason": forecast_row.get("blend_reason"),
            "shadow_p_blend": counterfactual.p_blend,
            "shadow_market_weight": counterfactual.market_weight,
            "shadow_model_weight": counterfactual.model_weight,
            "shadow_sigma_blend": counterfactual.sigma_blend,
            "shadow_blend_reason": counterfactual.blend_reason,
            "delta_p_blend": delta,
            "p_market": counterfactual.p_market,
            "p_cal": float(p_cal),
            "sigma_model": float(sigma_model),
        }
        rows.append(_shadow_envelope(
            ctx=ctx,
            market_id=market_id,
            variant_name=SHADOW_DEFERENCE,
            edge_source=SHADOW_DEFERENCE,
            payload=payload,
            p_mean=counterfactual.p_blend,
            sigma_p=counterfactual.sigma_blend,
        ))
    return rows


def build_confidence_shadow_rows(
    *,
    ctx: TickContext,
    features_by_market: dict[str, QuoteFeatures],
    forecast_audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One ``quote_history_confidence`` shadow per forecasted market.

    Records the A.11.2 channel-1 confidence modifier value that would
    multiply the A.5 score at ``confidence_modifier_weight=1.0``. The
    actual A.5 score lives in ``proposals.score``; analysts can join.
    """
    rows: list[dict[str, Any]] = []
    for forecast_row in forecast_audit_rows:
        if forecast_row.get("error"):
            continue
        if forecast_row.get("decision") not in ("ok", "cached"):
            continue
        market_id = forecast_row.get("market_id")
        features = features_by_market.get(market_id)
        if features is None:
            continue
        modifier_full = _confidence_modifier_raw(features)
        payload = {
            "variant": SHADOW_CONFIDENCE,
            "confidence_modifier_weight": 1.0,
            "shadow_qh_confidence_modifier": modifier_full,
            "baseline_qh_confidence_modifier": features.qh_confidence_modifier,
            "delta_modifier": modifier_full - features.qh_confidence_modifier,
            "quote_stability": features.quote_stability,
            "volatility_8_tick": features.volatility_8_tick,
            "spread_change_4_tick": features.spread_change_4_tick,
            "qh_warmup": features.qh_warmup,
        }
        rows.append(_shadow_envelope(
            ctx=ctx,
            market_id=market_id,
            variant_name=SHADOW_CONFIDENCE,
            edge_source=SHADOW_CONFIDENCE,
            payload=payload,
            score=modifier_full,
        ))
    return rows
