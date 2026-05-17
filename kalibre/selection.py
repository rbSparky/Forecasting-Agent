"""Phase 4B: deterministic forecast-target selector.

The Sonnet budget is small, so we cannot afford to forecast the first
``max_markets`` universe-accepted markets blindly. This module ranks
universe-accepted markets by their expected forecasting value, applies
AGENTS.md A.8 NO-LLM skip rules, and emits one audit row per
universe-accepted candidate so we can later compare "what we picked"
against "what we ignored".

Pure deterministic logic only. No I/O, no LLM, no randomness.

Phase 6: when the normal pass returns zero selected markets, an opt-in
exploration mode (:class:`SelectorExplorationConfig`) can promote up to
``exploration_max`` markets that would otherwise be skipped as
``skip_efficient_market``. The Sonnet/Opus probe lets us learn whether
the so-called efficient pricing is actually tight or merely consensus-by-
default. Exploration is disabled by default and never relaxes any other
gate (malformed quote, extreme pricing, short/long horizon, etc.).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable

from kalibre.layers import LayersConfig
from kalibre.tick_context import MarketView, PortfolioView, TickContext


# --- skip / accept reasons (stable strings; analytics expects these) -------

SKIP_INVALID_QUOTE = "skip_invalid_quote"
SKIP_EXTREME_PRICING = "skip_extreme_pricing"
SKIP_NEAR_RESOLUTION = "skip_near_resolution"
# Phase 6 note: under the default universe (24h floor, 720h ceiling)
# ``SKIP_SHORT_HORIZON`` and ``SKIP_LONG_HORIZON`` are dead paths -- the
# universe filter has already rejected those markets. The labels are
# retained because the selector is callable against custom universe
# configs (tests, future relaxations); when that happens they emit valid
# routing metadata. Do not duplicate them as hard rejects.
SKIP_SHORT_HORIZON = "skip_short_horizon"
SKIP_LONG_HORIZON = "skip_long_horizon"
SKIP_EFFICIENT_MARKET = "skip_efficient_market"
SKIP_DEGRADED_MODE = "skip_degraded_mode"
SKIP_DIVERSIFICATION_CAP = "skip_diversification_cap"
SKIP_BUDGET_FULL = "skip_budget_full"

ACCEPTED = "accepted"
ACCEPTED_ACTIVE_POSITION = "accepted_active_position"
# Phase 6: exploration-fallback promotions. The audit row preserves the
# original (overridden) skip reason via ``original_skip_reason`` so analysts
# can split realised PnL of exploration trades from normal-edge trades.
ACCEPTED_EXPLORATION = "selected_efficient_exploration"


# --- thresholds (mirrors AGENTS.md A.8 and B; documented + overridable) ----

PRICE_BAND_LO = 0.20
PRICE_BAND_HI = 0.80
EXTREME_ASK_MAX = 0.08
EXTREME_BID_MIN = 0.92
EFFICIENT_SPREAD_MAX = 0.02
EFFICIENT_VOLUME_MIN = 5000.0
NEAR_RESOLUTION_HOURS = 2.0
SHORT_HORIZON_HOURS = 6.0
# Phase 6: aligned to the 30-day SDK ceiling (UniverseFilterConfig default).
# Under the default universe (24h-720h), SHORT_HORIZON and LONG_HORIZON are
# dead paths -- the universe filter has already rejected anything outside
# the band. They remain valid metadata when the selector is invoked
# against a custom universe (tests, future widenings).
LONG_HORIZON_HOURS = 30.0 * 24.0
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
    def exploration_promotion_ids(self) -> list[str]:
        """Phase 6: ids the exploration fallback promoted from
        ``skip_efficient_market``. Stable order."""
        return [
            d.market_id
            for d in self.decisions
            if d.accepted and d.selection_reason == ACCEPTED_EXPLORATION
        ]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "selected_count": len(self.selected),
            "evaluated_count": len(self.decisions),
            "skip_reason_counts": self.skip_reason_counts,
            "selected_market_ids": self.selected_market_ids,
            "exploration_promotion_ids": self.exploration_promotion_ids,
        }


# --- Phase 6: exploration mode --------------------------------------------


DEFAULT_EXPLORATION_MAX = 3


@dataclass(frozen=True)
class SelectorExplorationConfig:
    """Phase 6 exploration fallback: promote efficient-market candidates when
    the normal selector returns nothing.

    Disabled by default. Reads three env vars via :meth:`from_env`:
    ``KALIBRE_SELECTOR_EXPLORATION_MODE`` (``0|1``),
    ``KALIBRE_SELECTOR_EXPLORATION_MAX`` (default 3),
    ``KALIBRE_SELECTOR_EXPLORATION_REQUIRE_ZERO_SELECTED`` (default 1).
    """

    enabled: bool = False
    exploration_max: int = DEFAULT_EXPLORATION_MAX
    require_zero_selected: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SelectorExplorationConfig":
        env_map = env if env is not None else dict(os.environ)
        try:
            exploration_max = int(
                env_map.get("KALIBRE_SELECTOR_EXPLORATION_MAX")
                or DEFAULT_EXPLORATION_MAX,
            )
        except ValueError:
            exploration_max = DEFAULT_EXPLORATION_MAX
        require_zero = (
            env_map.get("KALIBRE_SELECTOR_EXPLORATION_REQUIRE_ZERO_SELECTED", "1").strip()
            != "0"
        )
        return cls(
            enabled=(env_map.get("KALIBRE_SELECTOR_EXPLORATION_MODE", "").strip() == "1"),
            exploration_max=max(0, exploration_max),
            require_zero_selected=require_zero,
        )


def _exploration_score(audit: dict[str, Any]) -> float:
    """Phase 6 ranking helper for the exploration pool.

    Favours informative forecasting:
    - high volume (deep market => stable quote, meaningful prior);
    - tight (but not necessarily zero) spread => liquid;
    - mid-price near 0.5 => maximum information from any forecast edge.
    Each term in [0,1]. Sum stays bounded; ties broken by market_id.
    """
    vol = float(audit.get("volume_24h") or 0.0)
    spread = audit.get("spread")
    mid = audit.get("mid")
    vol_norm = 0.0
    if vol > 0:
        vol_norm = min(1.0, math.log10(max(1.0, vol)) / math.log10(VOLUME_HIGH))
    spread_inv = 0.0
    if spread is not None and spread >= 0:
        if spread <= 0.005:
            spread_inv = 1.0
        elif spread >= 0.05:
            spread_inv = 0.0
        else:
            spread_inv = max(0.0, 1.0 - (float(spread) - 0.005) / (0.05 - 0.005))
    mid_balance = 0.0
    if mid is not None:
        mid_balance = max(0.0, 1.0 - 2.0 * abs(float(mid) - 0.5))
    return vol_norm + 0.5 * mid_balance + 0.3 * spread_inv


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
    exploration_config: SelectorExplorationConfig | None = None,
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
       - ``time_to_resolve > 30d`` (capital lock too long; aligned to SDK ceiling);
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

    # --- Phase 6: exploration fallback -----------------------------------
    if exploration_config is not None and exploration_config.enabled:
        explore_now = (
            len(selected) == 0
            or not exploration_config.require_zero_selected
        )
        if explore_now and exploration_config.exploration_max > 0:
            (
                selected,
                decisions,
            ) = _apply_exploration_fallback(
                selected=selected,
                decisions=decisions,
                config=exploration_config,
                markets_by_id={m.market_id: m for m in markets_list},
            )

    return SelectionPassResult(selected=selected, decisions=decisions)


# --- Phase 6: exploration fallback helper ---------------------------------


def _apply_exploration_fallback(
    *,
    selected: list[MarketView],
    decisions: list[SelectionDecision],
    config: SelectorExplorationConfig,
    markets_by_id: dict[str, MarketView],
) -> tuple[list[MarketView], list[SelectionDecision]]:
    """Promote up to ``exploration_max`` efficient-market candidates.

    Pure rewriter on top of an existing :class:`SelectionPassResult`-style
    decisions list. Markets it promotes:

    - are already in the universe-accepted candidate set (i.e. appear in
      ``decisions`` with ``accepted=False`` and
      ``selection_reason == SKIP_EFFICIENT_MARKET``);
    - are not already selected (no double-counting);
    - rank highest by :func:`_exploration_score`, tie-broken by
      ``market_id`` for determinism;
    - subject to a soft topic/family diversification cap of 2 per family.

    Promoted decisions become ``accepted=True`` with
    ``selection_reason=ACCEPTED_EXPLORATION``; the original skip reason is
    preserved inside ``audit["original_skip_reason"]`` so downstream
    analytics never lose the fact that this was a fallback pick.
    """
    promotable: list[tuple[float, str, dict[str, Any]]] = []
    selected_ids = {m.market_id for m in selected}
    for i, decision in enumerate(decisions):
        if decision.accepted:
            continue
        if decision.selection_reason != SKIP_EFFICIENT_MARKET:
            continue
        if decision.market_id in selected_ids:
            continue
        score = _exploration_score(decision.audit)
        promotable.append((score, decision.market_id, decision.audit))
    promotable.sort(key=lambda x: (-x[0], x[1]))

    promoted: dict[str, tuple[float, dict[str, Any]]] = {}
    family_counts: dict[str, int] = {}
    for score, market_id, audit in promotable:
        if len(promoted) >= config.exploration_max:
            break
        family = str(audit.get("family") or audit.get("topic") or "_")
        if family_counts.get(family, 0) >= 2:
            continue
        promoted[market_id] = (score, audit)
        family_counts[family] = family_counts.get(family, 0) + 1

    if not promoted:
        return selected, decisions

    # Rewrite decisions in-place semantically: replace the original skip
    # row with a promoted "select" row that preserves the prior reason.
    new_decisions: list[SelectionDecision] = []
    for decision in decisions:
        if decision.market_id in promoted:
            score, _audit = promoted[decision.market_id]
            promoted_audit = {
                **decision.audit,
                "original_skip_reason": decision.selection_reason,
                "exploration_score": float(score),
            }
            new_decisions.append(SelectionDecision(
                market_id=decision.market_id,
                accepted=True,
                selection_score=float(score),
                selection_reason=ACCEPTED_EXPLORATION,
                rank=None,
                audit=promoted_audit,
            ))
        else:
            new_decisions.append(decision)

    # Append promoted markets to the selected list in deterministic order
    # (by exploration score desc, then market_id).
    promoted_markets = [
        markets_by_id[mid]
        for mid in sorted(
            promoted.keys(),
            key=lambda mid: (-promoted[mid][0], mid),
        )
        if mid in markets_by_id
    ]
    selected = list(selected) + promoted_markets
    return selected, new_decisions
