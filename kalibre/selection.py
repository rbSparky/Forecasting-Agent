"""Phase 4B: deterministic forecast-target selector.

The Sonnet budget is small, so we cannot afford to forecast the first
``max_markets`` universe-accepted markets blindly. This module ranks
universe-accepted markets by their expected forecasting value, applies
AGENTS.md A.8 NO-LLM skip rules, and emits one audit row per
universe-accepted candidate so we can later compare "what we picked"
against "what we ignored".

Pure deterministic logic only. No I/O, no LLM, no randomness.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable

from kalibre.layers import LayersConfig
from kalibre.tick_context import MarketView, PortfolioView, TickContext


# --- skip / accept reasons (stable strings; analytics expects these) -------

SKIP_INVALID_QUOTE = "skip_invalid_quote"
SKIP_EXTREME_PRICING = "skip_extreme_pricing"
SKIP_NEAR_RESOLUTION = "skip_near_resolution"
SKIP_SHORT_HORIZON = "skip_short_horizon"
SKIP_LONG_HORIZON = "skip_long_horizon"
SKIP_EFFICIENT_MARKET = "skip_efficient_market"
SKIP_DEGRADED_MODE = "skip_degraded_mode"
SKIP_DIVERSIFICATION_CAP = "skip_diversification_cap"
SKIP_BUDGET_FULL = "skip_budget_full"

ACCEPTED = "accepted"
ACCEPTED_ACTIVE_POSITION = "accepted_active_position"


# --- thresholds (mirrors AGENTS.md A.8 and B; documented + overridable) ----

PRICE_BAND_LO = 0.20
PRICE_BAND_HI = 0.80
EXTREME_ASK_MAX = 0.03
EXTREME_BID_MIN = 0.97
EFFICIENT_SPREAD_MAX = 0.01
EFFICIENT_VOLUME_MIN = 20000.0
NEAR_RESOLUTION_HOURS = 1.0
SHORT_HORIZON_HOURS = 2.0
LONG_HORIZON_HOURS = 365.0 * 24.0
MIN_VOLUME_FOR_SCORE = 200.0
VOLUME_HIGH = 200_000.0
SWEET_SPREAD_LO = 0.02
SWEET_SPREAD_HI = 0.04
WIDE_SPREAD_LIMIT = 0.10


# --- dataclasses ------------------------------------------------------------


@dataclass(frozen=True)
class SelectionDecision:
    """One audit row per universe-accepted candidate."""

    market_id: str
    accepted: bool
    selection_score: float
    selection_reason: str
    rank: int | None
    audit: dict[str, Any] = field(default_factory=dict)

    def to_shadow_row(
        self,
        *,
        ctx: TickContext,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        created = created_at or datetime.now(tz=UTC).isoformat()
        payload = {
            "variant": "forecast_selection",
            "market_id": self.market_id,
            "accepted": self.accepted,
            "selection_score": self.selection_score,
            "selection_reason": self.selection_reason,
            "rank": self.rank,
            **self.audit,
        }
        return {
            "variant_name": "forecast_selection",
            "tick_ts": ctx.tick_id,
            "experiment_id": ctx.experiment_id,
            "participant_idx": ctx.participant_idx,
            "market_id": self.market_id,
            "edge_source": "forecast_selection",
            "p_mean": None,
            "sigma_p": None,
            "side": None,
            "score": self.selection_score,
            "decision": "select" if self.accepted else "skip",
            "reject_reason": None if self.accepted else self.selection_reason,
            "audit_json": json.dumps(payload, default=str),
            "created_at": created,
        }


@dataclass(frozen=True)
class SelectionPassResult:
    selected: list[MarketView]
    decisions: list[SelectionDecision]

    @property
    def skip_reason_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            if d.accepted:
                continue
            out[d.selection_reason] = out.get(d.selection_reason, 0) + 1
        return out

    @property
    def selected_market_ids(self) -> list[str]:
        return [m.market_id for m in self.selected]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "selected_count": len(self.selected),
            "evaluated_count": len(self.decisions),
            "skip_reason_counts": self.skip_reason_counts,
            "selected_market_ids": self.selected_market_ids,
        }


# --- helpers ----------------------------------------------------------------


def _safe_score(score: float) -> float:
    if score != score:  # NaN
        return 0.0
    return max(0.0, score)


def _price_band_score(mid: float | None) -> float:
    if mid is None:
        return 0.0
    if PRICE_BAND_LO <= mid <= PRICE_BAND_HI:
        return 1.0
    if mid < PRICE_BAND_LO:
        return _safe_score(mid / PRICE_BAND_LO)
    return _safe_score((1.0 - mid) / PRICE_BAND_LO)


def _volume_score(volume_24h: float | None) -> float:
    if volume_24h is None or volume_24h <= 0:
        return 0.0
    vol = max(MIN_VOLUME_FOR_SCORE, min(VOLUME_HIGH, float(volume_24h)))
    span = math.log10(VOLUME_HIGH) - math.log10(MIN_VOLUME_FOR_SCORE)
    return (math.log10(vol) - math.log10(MIN_VOLUME_FOR_SCORE)) / span


def _spread_score(spread: float | None) -> float:
    if spread is None or spread < 0:
        return 0.0
    if SWEET_SPREAD_LO <= spread <= SWEET_SPREAD_HI:
        return 1.0
    if spread < SWEET_SPREAD_LO:
        # Ultra-tight markets are penalized (the efficient skip handles
        # the worst offenders; this captures the soft tail).
        return max(0.4, spread / SWEET_SPREAD_LO)
    if spread >= WIDE_SPREAD_LIMIT:
        return 0.0
    return max(0.0, 1.0 - (spread - SWEET_SPREAD_HI) / (WIDE_SPREAD_LIMIT - SWEET_SPREAD_HI))


def _horizon_score(ttl_hours: float | None) -> float:
    if ttl_hours is None or ttl_hours <= 0:
        return 0.0
    if 24.0 <= ttl_hours <= 24.0 * 7:
        return 1.0
    if ttl_hours < 24.0:
        return max(0.0, ttl_hours / 24.0)
    if ttl_hours > LONG_HORIZON_HOURS:
        return 0.0
    return max(0.3, 1.0 - (ttl_hours - 24.0 * 7) / (LONG_HORIZON_HOURS - 24.0 * 7))


def _topic_key(market: MarketView) -> tuple[str, str]:
    return (market.category or "other", market.family or "_")


def _smart_market_signal(qh_features: dict[str, Any] | None, market_id: str) -> bool:
    if not qh_features:
        return False
    f = qh_features.get(market_id)
    if f is None:
        return False
    return bool(getattr(f, "market_is_smart", False))


# --- selectors --------------------------------------------------------------


def passthrough_selector(
    *,
    ctx: TickContext,
    markets: Iterable[MarketView],
    max_markets: int,
    **_kwargs: Any,
) -> SelectionPassResult:
    """Legacy first-N truncation.

    Used by tests that want to bypass the production selector. Production
    callers should use :func:`select_forecast_targets`.
    """
    materialized = list(markets)
    selected = materialized[: max(0, int(max_markets))]
    return SelectionPassResult(selected=selected, decisions=[])


def select_forecast_targets(
    *,
    ctx: TickContext,
    markets: Iterable[MarketView],
    max_markets: int,
    portfolio: PortfolioView | None = None,
    qh_features: dict[str, Any] | None = None,
    layer_config: LayersConfig | None = None,
    degraded_mode: str = "full",
    max_per_topic: int | None = None,
) -> SelectionPassResult:
    """Deterministically pick up to ``max_markets`` LLM-worthy candidates.

    Returns the chosen markets plus one :class:`SelectionDecision` per
    *universe-accepted* candidate. Every decision carries
    ``selection_score`` and ``selection_reason``.

    Rules in order:

    1. Spend governor in ``structural_only`` mode -> skip everything.
    2. Per-market hard skips (A.8 NO-LLM zones):

       - malformed quote;
       - ``ask < 0.08`` / ``bid > 0.92`` (longshot layer);
       - ``time_to_resolve < 2h`` (market dominates LLM);
       - ``time_to_resolve <= 6h`` (too late to act on a 1-tick forecast);
       - ``time_to_resolve > 21d`` (capital lock too long);
       - ``spread <= 0.02 AND volume >= 5K`` (efficient market) **unless**
         we hold an active position on the market.

    3. The remaining candidates are scored on price-band fit, volume,
       spread sweet spot, and horizon; active positions get a +1 bonus
       and bypass the efficient-market skip + diversification cap.
    4. Topic+family diversification: at most
       ``max(1, max_markets // 3)`` markets per ``(topic, family)``
       bucket (unless the market is an active position).
    """
    markets_list = list(markets)
    decisions: list[SelectionDecision] = []
    held_market_ids: set[str] = set()
    if portfolio is not None:
        held_market_ids = {p.market_id for p in portfolio.positions if p.shares > 0}

    if degraded_mode == "structural_only":
        decisions.extend(
            SelectionDecision(
                market_id=m.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_DEGRADED_MODE,
                rank=None, audit={"degraded_mode": degraded_mode},
            )
            for m in markets_list
        )
        return SelectionPassResult(selected=[], decisions=decisions)

    cap = max(0, int(max_markets))
    # Diversification floor: with very small ``max_markets`` we don't want
    # to force-empty a tick. Default cap is ``max(2, max_markets // 3)``
    # so max=2 -> 2 (no cap), max=6 -> 2, max=10 -> 3.
    per_topic_cap = max_per_topic if max_per_topic is not None else max(2, cap // 3)

    scored: list[tuple[float, MarketView, dict[str, Any], bool]] = []
    for market in markets_list:
        q = market.quote
        mid = q.mid
        spread = q.spread
        volume = q.volume_24h
        ttl_hours = market.hours_to_resolution(ctx.now)
        active = market.market_id in held_market_ids
        common_audit = {
            "best_bid": q.best_bid,
            "best_ask": q.best_ask,
            "mid": mid,
            "spread": spread,
            "volume_24h": volume,
            "ttl_hours": ttl_hours,
            "active_position": active,
            "topic": market.category,
            "family": market.family,
        }

        if not q.is_well_formed or mid is None:
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_INVALID_QUOTE,
                rank=None, audit=common_audit,
            ))
            continue

        if not active and (
            (q.best_ask is not None and q.best_ask < EXTREME_ASK_MAX)
            or (q.best_bid is not None and q.best_bid > EXTREME_BID_MIN)
        ):
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_EXTREME_PRICING,
                rank=None, audit=common_audit,
            ))
            continue

        if ttl_hours is not None and ttl_hours < NEAR_RESOLUTION_HOURS:
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_NEAR_RESOLUTION,
                rank=None, audit=common_audit,
            ))
            continue

        if not active and (ttl_hours is None or ttl_hours <= SHORT_HORIZON_HOURS):
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_SHORT_HORIZON,
                rank=None, audit=common_audit,
            ))
            continue

        if not active and ttl_hours is not None and ttl_hours > LONG_HORIZON_HOURS:
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_LONG_HORIZON,
                rank=None, audit=common_audit,
            ))
            continue

        if (
            not active
            and spread is not None and spread <= EFFICIENT_SPREAD_MAX
            and volume is not None and volume >= EFFICIENT_VOLUME_MIN
        ):
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=0.0, selection_reason=SKIP_EFFICIENT_MARKET,
                rank=None, audit={**common_audit, "qh_smart": _smart_market_signal(qh_features, market.market_id)},
            ))
            continue

        # Score eligible candidates.
        price_score = _price_band_score(mid)
        volume_score = _volume_score(volume)
        spread_score = _spread_score(spread)
        horizon_score = _horizon_score(ttl_hours)
        active_bonus = 1.0 if active else 0.0
        selection_score = price_score + volume_score + spread_score + horizon_score + active_bonus
        audit = {
            **common_audit,
            "price_score": price_score,
            "volume_score": volume_score,
            "spread_score": spread_score,
            "horizon_score": horizon_score,
            "active_bonus": active_bonus,
        }
        scored.append((selection_score, market, audit, active))

    # Deterministic ordering: descending score, then market_id for ties.
    scored.sort(key=lambda x: (-x[0], x[1].market_id))

    selected: list[MarketView] = []
    selected_decisions: list[SelectionDecision] = []
    topic_counts: dict[tuple[str, str], int] = {}
    unselected: list[tuple[float, MarketView, dict[str, Any]]] = []

    for score, market, audit, active in scored:
        if len(selected) >= cap:
            unselected.append((score, market, audit))
            continue
        topic_key = _topic_key(market)
        if not active and topic_counts.get(topic_key, 0) >= per_topic_cap:
            decisions.append(SelectionDecision(
                market_id=market.market_id, accepted=False,
                selection_score=score, selection_reason=SKIP_DIVERSIFICATION_CAP,
                rank=None,
                audit={**audit, "topic_key": list(topic_key), "topic_cap": per_topic_cap},
            ))
            continue
        rank = len(selected) + 1
        reason = ACCEPTED_ACTIVE_POSITION if active else ACCEPTED
        selected.append(market)
        topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
        selected_decisions.append(SelectionDecision(
            market_id=market.market_id, accepted=True,
            selection_score=score, selection_reason=reason,
            rank=rank, audit=audit,
        ))

    # Mark the score-eligible but cap-displaced markets.
    for score, market, audit in unselected:
        decisions.append(SelectionDecision(
            market_id=market.market_id, accepted=False,
            selection_score=score, selection_reason=SKIP_BUDGET_FULL,
            rank=None, audit=audit,
        ))

    decisions.extend(selected_decisions)
    return SelectionPassResult(selected=selected, decisions=decisions)
