"""Typed snapshots used by the Phase 2 deterministic risk engine.

The Prophet Arena SDK returns ``MarketData`` and ``PortfolioResponse``
with quote / price / share fields as *strings*. Strategy modules want
floats, derived exposures, and quote age in seconds. This module is the
single place where SDK shapes are normalized for internal consumption.

Everything here is deterministic, side-effect-free, and easy to test.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


def _to_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, or return ``None`` if impossible."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int; reject.
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    if isinstance(value, Decimal):
        with contextlib.suppress(InvalidOperation, ValueError):
            return float(value)
        return None
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        with contextlib.suppress(InvalidOperation, ValueError):
            return float(Decimal(v))
    return None


def _ensure_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


@dataclass(frozen=True)
class QuoteView:
    """Normalized quote with derived fields."""

    best_bid: float | None
    best_ask: float | None
    volume_24h: float | None
    ts: datetime | None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def quote_age_sec(self, now: datetime) -> float | None:
        if self.ts is None:
            return None
        return (now - self.ts).total_seconds()

    @property
    def is_well_formed(self) -> bool:
        """True iff bid/ask are present and form a sane quote."""
        if self.best_bid is None or self.best_ask is None:
            return False
        if not (0.0 < self.best_bid < 1.0):
            return False
        if not (0.0 < self.best_ask < 1.0):
            return False
        if self.best_ask < self.best_bid:
            return False
        return True


@dataclass(frozen=True)
class MarketView:
    """Strategy-side view of a candidate market."""

    market_id: str
    question: str
    source: str | None
    topic: str | None
    family: str | None
    resolution_time: datetime | None
    quote: QuoteView

    def hours_to_resolution(self, now: datetime) -> float | None:
        if self.resolution_time is None:
            return None
        return (self.resolution_time - now).total_seconds() / 3600.0

    @property
    def category(self) -> str:
        return (self.topic or "other").lower()


def market_view_from_sdk(market: Any) -> MarketView:
    """Build a :class:`MarketView` from an SDK ``MarketData``-shaped object."""
    quote_obj = getattr(market, "quote", None)
    quote = QuoteView(
        best_bid=_to_float(getattr(quote_obj, "best_bid", None)),
        best_ask=_to_float(getattr(quote_obj, "best_ask", None)),
        volume_24h=_to_float(getattr(quote_obj, "volume_24h", None)),
        ts=_ensure_utc(getattr(quote_obj, "ts", None)) if quote_obj is not None else None,
    )
    return MarketView(
        market_id=str(getattr(market, "market_id", "") or ""),
        question=str(getattr(market, "question", "") or ""),
        source=getattr(market, "source", None) or None,
        topic=getattr(market, "topic", None) or None,
        family=getattr(market, "family", None) or None,
        resolution_time=_ensure_utc(getattr(market, "resolution_time", None)),
        quote=quote,
    )


@dataclass(frozen=True)
class PositionView:
    """Strategy-side view of an open position."""

    market_id: str
    side: str
    shares: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    updated_at: datetime | None

    @property
    def notional_usd(self) -> float:
        """Mark-to-market notional in dollars."""
        return self.shares * self.current_price


def position_view_from_sdk(position: Any) -> PositionView:
    return PositionView(
        market_id=str(getattr(position, "market_id", "") or ""),
        side=str(getattr(position, "side", "") or "").upper(),
        shares=_to_float(getattr(position, "shares", None)) or 0.0,
        avg_entry_price=_to_float(getattr(position, "avg_entry_price", None)) or 0.0,
        current_price=_to_float(getattr(position, "current_price", None)) or 0.0,
        unrealized_pnl=_to_float(getattr(position, "unrealized_pnl", None)) or 0.0,
        realized_pnl=_to_float(getattr(position, "realized_pnl", None)) or 0.0,
        updated_at=_ensure_utc(getattr(position, "updated_at", None)),
    )


@dataclass(frozen=True)
class PortfolioView:
    """Strategy-side portfolio with derived exposure aggregates."""

    cash: float
    equity: float
    total_pnl: float
    positions: tuple[PositionView, ...]
    total_fills: int

    # --- derived aggregates ------------------------------------------------

    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions if p.shares > 0)

    @property
    def gross_exposure_usd(self) -> float:
        return sum(max(0.0, p.notional_usd) for p in self.positions)

    def exposure_by_market(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.market_id] = out.get(p.market_id, 0.0) + max(0.0, p.notional_usd)
        return out

    def side_by_market(self) -> dict[str, str]:
        """The currently held side per market. Empty if no holdings."""
        out: dict[str, str] = {}
        for p in self.positions:
            if p.shares > 0:
                out[p.market_id] = p.side
        return out

    def exposure_by_topic(self, market_by_id: dict[str, MarketView]) -> dict[str, float]:
        """Aggregate notional exposure per topic/category.

        ``market_by_id`` provides the topic mapping for the markets we still
        hold. Markets that have dropped out of the universe contribute to the
        ``"unknown"`` bucket.
        """
        out: dict[str, float] = {}
        for p in self.positions:
            cat = (market_by_id[p.market_id].category if p.market_id in market_by_id else "unknown")
            out[cat] = out.get(cat, 0.0) + max(0.0, p.notional_usd)
        return out


def portfolio_view_from_sdk(portfolio: Any) -> PortfolioView:
    if portfolio is None:
        return PortfolioView(cash=0.0, equity=0.0, total_pnl=0.0, positions=(), total_fills=0)
    positions_raw = getattr(portfolio, "positions", None) or ()
    positions = tuple(position_view_from_sdk(p) for p in positions_raw)
    return PortfolioView(
        cash=_to_float(getattr(portfolio, "cash", None)) or 0.0,
        equity=_to_float(getattr(portfolio, "equity", None)) or 0.0,
        total_pnl=_to_float(getattr(portfolio, "total_pnl", None)) or 0.0,
        positions=positions,
        total_fills=int(getattr(portfolio, "total_fills", 0) or 0),
    )


@dataclass(frozen=True)
class TickContext:
    """Everything Phase 2 strategy modules need to make decisions."""

    tick_id: str
    tick_ts: datetime
    candidate_set_id: str | None
    markets: tuple[MarketView, ...]
    portfolio: PortfolioView
    now: datetime
    experiment_id: str
    participant_idx: int
    version: str
    spend_governor_state: dict[str, Any] = field(default_factory=dict)
    pnl_24h: float = 0.0

    def market_by_id(self) -> dict[str, MarketView]:
        return {m.market_id: m for m in self.markets}


def build_tick_context(
    *,
    tick_id: str,
    tick_ts: datetime,
    candidate_set_id: str | None,
    markets: Iterable[Any],
    portfolio: Any,
    experiment_id: str,
    participant_idx: int,
    version: str,
    now: datetime | None = None,
    pnl_24h: float = 0.0,
    spend_governor_state: dict[str, Any] | None = None,
) -> TickContext:
    """Assemble a :class:`TickContext` from raw SDK objects."""
    return TickContext(
        tick_id=tick_id,
        tick_ts=_ensure_utc(tick_ts) or datetime.now(tz=UTC),
        candidate_set_id=candidate_set_id,
        markets=tuple(market_view_from_sdk(m) for m in markets),
        portfolio=portfolio_view_from_sdk(portfolio),
        now=_ensure_utc(now) or datetime.now(tz=UTC),
        experiment_id=experiment_id,
        participant_idx=participant_idx,
        version=version,
        spend_governor_state=dict(spend_governor_state or {}),
        pnl_24h=pnl_24h,
    )
