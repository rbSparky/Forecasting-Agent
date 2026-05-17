"""Layer 1 deterministic universe filter.

Rejects candidate markets that are unsafe to trade for *structural*
reasons (volume / spread / quote-age / horizon / malformed quote). The
filter is pure: same input -> same output, no I/O.

Thresholds default to the values listed in ``AGENTS.md`` Section B but
are passed as a :class:`UniverseFilterConfig` so tests can override
without touching production constants.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from kalibre.tick_context import MarketView


# Reject reasons used in audit rows. Stable strings; new reasons must be
# added explicitly so audits stay machine-readable.
REJECT_LOW_VOLUME = "universe_low_volume"
REJECT_MALFORMED_QUOTE = "universe_malformed_quote"
REJECT_WIDE_SPREAD_KALSHI = "universe_wide_spread_kalshi"
REJECT_WIDE_SPREAD_POLYMARKET = "universe_wide_spread_polymarket"
REJECT_WIDE_SPREAD_UNKNOWN_SOURCE = "universe_wide_spread_unknown_source"
REJECT_STALE_QUOTE = "universe_stale_quote"
REJECT_RESOLUTION_TOO_SOON = "universe_resolution_too_soon"
REJECT_RESOLUTION_TOO_FAR = "universe_resolution_too_far"
REJECT_MISSING_RESOLUTION_TIME = "universe_missing_resolution_time"

# Phase 6: shadow row tag for markets that would pass every gate except the
# 30-day horizon cap. Persisted as a `shadow_proposals` row by the loop;
# the universe filter never trades these, so they have no effect on live
# behaviour. Lets the operator see the would-be wider pool.
SHADOW_HORIZON_EXTENDED = "universe_horizon_extended"


@dataclass(frozen=True)
class UniverseFilterConfig:
    """Tunable thresholds for the universe filter."""

    min_volume_24h_usd: float = 200.0
    max_spread_kalshi: float = 0.05
    max_spread_polymarket: float = 0.03
    max_spread_fallback: float = 0.05
    max_quote_age_sec: float = 600.0
    min_hours_to_resolution: float = 24.0
    # Phase 6: bumped from 21d to 30d to match the SDK ruleset
    # (ai_prophet_core.ruleset.MAX_HOURS_TO_RESOLUTION = 720).
    max_hours_to_resolution: float = 30.0 * 24.0
    # Phase 6 diagnostic-only: hours beyond which a market also fails the
    # horizon-extended shadow stream. 90 days is wide enough to capture
    # the next regime (election markets) without flooding the report.
    extended_diagnostic_max_hours: float = 90.0 * 24.0


@dataclass(frozen=True)
class UniverseDecision:
    """Per-market decision with a reject reason if applicable."""

    market_id: str
    accepted: bool
    reject_reason: str | None
    spread: float | None
    volume_24h: float | None
    quote_age_sec: float | None
    hours_to_resolution: float | None
    source: str | None


Decision = Literal["accept", "reject"]


def evaluate_market(
    market: MarketView,
    *,
    now: datetime,
    config: UniverseFilterConfig = UniverseFilterConfig(),
) -> UniverseDecision:
    """Apply the filter to a single market. First failed check wins."""
    q = market.quote
    spread = q.spread
    age = q.quote_age_sec(now)
    hours = market.hours_to_resolution(now)

    if not q.is_well_formed:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_MALFORMED_QUOTE,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )

    if q.volume_24h is None or q.volume_24h < config.min_volume_24h_usd:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_LOW_VOLUME,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )

    source = (market.source or "").lower()
    if source == "kalshi":
        max_spread = config.max_spread_kalshi
        reason = REJECT_WIDE_SPREAD_KALSHI
    elif source == "polymarket":
        max_spread = config.max_spread_polymarket
        reason = REJECT_WIDE_SPREAD_POLYMARKET
    else:
        max_spread = config.max_spread_fallback
        reason = REJECT_WIDE_SPREAD_UNKNOWN_SOURCE
    if spread is None or spread > max_spread:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=reason,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )

    if age is None or age > config.max_quote_age_sec:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_STALE_QUOTE,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )

    if hours is None:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_MISSING_RESOLUTION_TIME,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )
    if hours < config.min_hours_to_resolution:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_RESOLUTION_TOO_SOON,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )
    if hours > config.max_hours_to_resolution:
        return UniverseDecision(
            market_id=market.market_id, accepted=False,
            reject_reason=REJECT_RESOLUTION_TOO_FAR,
            spread=spread, volume_24h=q.volume_24h,
            quote_age_sec=age, hours_to_resolution=hours,
            source=market.source,
        )

    return UniverseDecision(
        market_id=market.market_id, accepted=True,
        reject_reason=None,
        spread=spread, volume_24h=q.volume_24h,
        quote_age_sec=age, hours_to_resolution=hours,
        source=market.source,
    )


def filter_universe(
    markets: Iterable[MarketView],
    *,
    now: datetime,
    config: UniverseFilterConfig = UniverseFilterConfig(),
) -> tuple[tuple[MarketView, ...], tuple[UniverseDecision, ...]]:
    """Return ``(accepted, decisions)`` for the supplied markets.

    ``decisions`` has one entry per input market in input order so audit
    code can correlate rejections.
    """
    accepted: list[MarketView] = []
    decisions: list[UniverseDecision] = []
    by_id = {m.market_id: m for m in markets}
    # Iterate once for stability.
    for market in by_id.values():
        decision = evaluate_market(market, now=now, config=config)
        decisions.append(decision)
        if decision.accepted:
            accepted.append(market)
    return tuple(accepted), tuple(decisions)


def horizon_extended_shadow_rows(
    markets: Iterable[MarketView],
    *,
    now: datetime,
    config: UniverseFilterConfig = UniverseFilterConfig(),
) -> tuple[UniverseDecision, ...]:
    """Phase 6 diagnostic: markets that pass every check **except** the
    horizon cap, within ``(max_hours_to_resolution, extended_diagnostic_max_hours]``.

    Returns a tuple of :class:`UniverseDecision` with ``accepted=False`` and
    ``reject_reason=SHADOW_HORIZON_EXTENDED``. These rows are persisted as a
    separate ``shadow_proposals`` stream by the loop and never reach the
    selector or the strategy. They exist so the operator can see how much
    of the would-be wider pool we are choosing not to trade.
    """
    if config.extended_diagnostic_max_hours <= config.max_hours_to_resolution:
        return ()
    # Re-evaluate with a temporarily wider horizon to find the ones that
    # would pass on every *other* gate.
    wide_config = UniverseFilterConfig(
        min_volume_24h_usd=config.min_volume_24h_usd,
        max_spread_kalshi=config.max_spread_kalshi,
        max_spread_polymarket=config.max_spread_polymarket,
        max_spread_fallback=config.max_spread_fallback,
        max_quote_age_sec=config.max_quote_age_sec,
        min_hours_to_resolution=config.min_hours_to_resolution,
        max_hours_to_resolution=config.extended_diagnostic_max_hours,
        extended_diagnostic_max_hours=config.extended_diagnostic_max_hours,
    )
    extended: list[UniverseDecision] = []
    seen: set[str] = set()
    by_id = {m.market_id: m for m in markets}
    for market in by_id.values():
        if market.market_id in seen:
            continue
        seen.add(market.market_id)
        # Skip anything that would already pass the strict filter --
        # those land in `accepted` and are not "extended" candidates.
        strict_decision = evaluate_market(market, now=now, config=config)
        if strict_decision.accepted:
            continue
        # Re-evaluate with the wider horizon: if it still fails, this is
        # not a horizon-bound market and we don't shadow it.
        wide_decision = evaluate_market(market, now=now, config=wide_config)
        if not wide_decision.accepted:
            continue
        extended.append(UniverseDecision(
            market_id=market.market_id,
            accepted=False,
            reject_reason=SHADOW_HORIZON_EXTENDED,
            spread=wide_decision.spread,
            volume_24h=wide_decision.volume_24h,
            quote_age_sec=wide_decision.quote_age_sec,
            hours_to_resolution=wide_decision.hours_to_resolution,
            source=wide_decision.source,
        ))
    return tuple(extended)
