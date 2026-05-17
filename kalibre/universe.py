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


@dataclass(frozen=True)
class UniverseFilterConfig:
    """Tunable thresholds for the universe filter."""

    min_volume_24h_usd: float = 50.0
    max_spread_kalshi: float = 0.12
    max_spread_polymarket: float = 0.10
    max_spread_fallback: float = 0.12
    max_quote_age_sec: float = 600.0
    min_hours_to_resolution: float = 2.0
    max_hours_to_resolution: float = 365.0 * 24.0


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
