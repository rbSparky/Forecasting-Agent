"""Live-canary safety clip.

Phase 4B adds a final clip after the strategy allocator but before
``submit_intents``. The clip is gated by ``KALIBRE_LIVE_CANARY_MODE=1``
and tightens the number of submitted intents and the per-intent
dollar exposure. It does not enable live trading -- a non-empty
intent list still requires ``KALIBRE_ENABLE_LIVE_TRADES=1`` to make it
out of the loop.

The module is pure deterministic logic. Tests exercise it directly via
:func:`apply_canary_clip`; the loop hooks it after the strategy result
in canary mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from ai_prophet_core import TradeIntentRequest

from kalibre.tick_context import MarketView


DEFAULT_CANARY_MAX_INTENTS = 1
DEFAULT_CANARY_MAX_SIZE_USD = 50.0


def _bool_env(env: dict[str, str], key: str) -> bool:
    return (env.get(key) or "").strip() == "1"


@dataclass(frozen=True)
class CanaryConfig:
    enabled: bool = False
    max_intents: int = DEFAULT_CANARY_MAX_INTENTS
    max_size_usd: float = DEFAULT_CANARY_MAX_SIZE_USD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "CanaryConfig":
        env = env if env is not None else dict(os.environ)
        try:
            max_intents = int(env.get("KALIBRE_CANARY_MAX_INTENTS") or DEFAULT_CANARY_MAX_INTENTS)
        except ValueError:
            max_intents = DEFAULT_CANARY_MAX_INTENTS
        try:
            max_size_usd = float(env.get("KALIBRE_CANARY_MAX_SIZE_USD") or DEFAULT_CANARY_MAX_SIZE_USD)
        except ValueError:
            max_size_usd = DEFAULT_CANARY_MAX_SIZE_USD
        return cls(
            enabled=_bool_env(env, "KALIBRE_LIVE_CANARY_MODE"),
            max_intents=max(0, max_intents),
            max_size_usd=max(0.0, max_size_usd),
        )


@dataclass(frozen=True)
class CanaryClipEntry:
    market_id: str
    original_shares: float
    clipped_shares: float
    price: float
    original_size_usd: float
    clipped_size_usd: float


@dataclass(frozen=True)
class CanaryDroppedIntent:
    """One submitted-but-removed intent recorded for provenance preservation."""

    market_id: str
    action: str
    side: str
    shares: float | None


@dataclass
class CanaryClipReport:
    enabled: bool
    intents_before: int
    intents_after: int
    removed_by_count: int
    size_clipped: list[CanaryClipEntry] = field(default_factory=list)
    dropped: list[CanaryDroppedIntent] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "intents_before": self.intents_before,
            "intents_after": self.intents_after,
            "removed_by_count": self.removed_by_count,
            "size_clipped": [
                {
                    "market_id": e.market_id,
                    "original_shares": e.original_shares,
                    "clipped_shares": e.clipped_shares,
                    "price": e.price,
                    "original_size_usd": e.original_size_usd,
                    "clipped_size_usd": e.clipped_size_usd,
                }
                for e in self.size_clipped
            ],
            "dropped": [
                {
                    "market_id": d.market_id,
                    "action": d.action,
                    "side": d.side,
                    "shares": d.shares,
                }
                for d in self.dropped
            ],
        }


def _format_shares(n: float) -> str:
    if n <= 0:
        return "0"
    rounded = round(n, 4)
    if rounded <= 0:
        return "0"
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _implied_price(market: MarketView | None, side: str) -> float | None:
    """Return ``best_ask`` for YES or ``1 - best_bid`` for NO."""
    if market is None:
        return None
    q = market.quote
    if side.upper() == "YES":
        if q.best_ask is None:
            return None
        return float(q.best_ask)
    if side.upper() == "NO":
        if q.best_bid is None:
            return None
        return max(1e-9, 1.0 - float(q.best_bid))
    return None


def apply_canary_clip(
    intents: Iterable[TradeIntentRequest],
    *,
    config: CanaryConfig,
    markets_by_id: dict[str, MarketView],
) -> tuple[list[TradeIntentRequest], CanaryClipReport]:
    """Trim the intent list under canary caps.

    Returns ``(intents_after, report)``. When ``config.enabled`` is
    False, intents are returned unchanged and the report reflects that
    no clipping was performed. The function never raises -- malformed
    intents pass through untouched and surface via the report.
    """
    materialized = list(intents)
    if not config.enabled:
        return materialized, CanaryClipReport(
            enabled=False,
            intents_before=len(materialized),
            intents_after=len(materialized),
            removed_by_count=0,
        )

    initial = len(materialized)
    cap = max(0, int(config.max_intents))
    truncated = materialized[:cap]
    dropped_intents = materialized[cap:]
    removed = max(0, initial - cap)
    final: list[TradeIntentRequest] = []
    size_clipped: list[CanaryClipEntry] = []
    dropped: list[CanaryDroppedIntent] = []
    for d in dropped_intents:
        try:
            shares_val: float | None = float(d.shares)
        except (TypeError, ValueError):
            shares_val = None
        dropped.append(CanaryDroppedIntent(
            market_id=d.market_id, action=d.action, side=d.side, shares=shares_val,
        ))
    for intent in truncated:
        price = _implied_price(markets_by_id.get(intent.market_id), intent.side)
        try:
            original_shares = float(intent.shares)
        except (TypeError, ValueError):
            final.append(intent)
            continue
        if price is None or price <= 0:
            final.append(intent)
            continue
        original_size = original_shares * price
        if original_size <= config.max_size_usd + 1e-9:
            final.append(intent)
            continue
        clipped_shares = config.max_size_usd / price
        final.append(TradeIntentRequest(
            market_id=intent.market_id,
            action=intent.action,
            side=intent.side,
            shares=_format_shares(clipped_shares),
            idempotency_key=intent.idempotency_key,
        ))
        size_clipped.append(CanaryClipEntry(
            market_id=intent.market_id,
            original_shares=original_shares,
            clipped_shares=clipped_shares,
            price=price,
            original_size_usd=original_size,
            clipped_size_usd=clipped_shares * price,
        ))
    return final, CanaryClipReport(
        enabled=True,
        intents_before=initial,
        intents_after=len(final),
        removed_by_count=removed,
        size_clipped=size_clipped,
        dropped=dropped,
    )
