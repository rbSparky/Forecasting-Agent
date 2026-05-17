"""Favorite/longshot structural prior (AGENTS.md A.6) -- shadow-only by default.

A market is a longshot/favorite candidate when ``yes_ask < 0.10`` or
``yes_bid > 0.90``. The prior is::

    p_prior = clip(p_market * alpha_category, 0.02, 0.98)

Where ``alpha_category`` comes from the committed
``forecast_config.toml`` ``[layers.longshot.alpha]`` table.

Gating (all must pass before the candidate becomes eligible):

- ``time_to_resolve <= max_time_to_resolve_days`` (default 7).
- ``open_longshots < max_open_longshots`` (default 5).
- The market's category is not currently in stop-loss state.

Stop-loss:

- Maintain a rolling window of the last ``stop_loss_window`` (default 20)
  resolved outcomes per (category, extreme_side).
- If observed win rate falls below ``stop_loss_win_rate_threshold``
  (default 0.40), suspend the category for
  ``stop_loss_suspension_hours`` hours.

Phase 4A persists a :class:`LongshotShadowRow` per candidate (eligible
*and* blocked) into ``shadow_proposals`` with ``variant_name='longshot'``.
Promotion to the primary probability dict is gated by
``LongshotLayer.primary_enabled``, which defaults to False.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Iterator

from kalibre.forecast.blend import P_MAX, P_MIN, compute_market_prior
from kalibre.layers import LongshotLayer
from kalibre.tick_context import MarketView, PortfolioView, TickContext


logger = logging.getLogger("kalibre.longshot")


LONGSHOT_SIDE = "longshot"
FAVORITE_SIDE = "favorite"


LONGSHOT_ASK_MAX = 0.10
FAVORITE_BID_MIN = 0.90


REJECT_NOT_EXTREME = "not_extreme_pricing"
REJECT_NO_MARKET_PRIOR = "no_market_prior"
REJECT_TIME_HORIZON = "time_horizon_too_far"
REJECT_OPEN_CAP = "open_longshot_cap"
REJECT_STOP_LOSS = "category_stop_loss"
DECISION_ELIGIBLE = "eligible"
DECISION_BLOCKED = "blocked"


def _clip(value: float, lo: float, hi: float) -> float:
    if value != value:  # NaN
        return (lo + hi) / 2.0
    return min(hi, max(lo, value))


def classify_extreme_side(market: MarketView) -> str | None:
    """Return ``'longshot'`` / ``'favorite'`` / ``None``."""
    q = market.quote
    if q.best_ask is not None and q.best_ask < LONGSHOT_ASK_MAX:
        return LONGSHOT_SIDE
    if q.best_bid is not None and q.best_bid > FAVORITE_BID_MIN:
        return FAVORITE_SIDE
    return None


# --- stop-loss state ------------------------------------------------------


@dataclass(frozen=True)
class CategoryState:
    category: str
    extreme_side: str
    n_resolved_last20: int
    n_wins_last20: int
    win_rate_last20: float | None
    stop_loss_until: datetime | None

    def in_stop_loss(self, *, now: datetime) -> bool:
        if self.stop_loss_until is None:
            return False
        return now < self.stop_loss_until


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _read_window(
    conn: sqlite3.Connection,
    *,
    category: str,
    extreme_side: str,
    window: int,
) -> tuple[int, int]:
    rows = list(conn.execute(
        "SELECT result FROM longshot_outcomes "
        "WHERE category=? AND extreme_side=? "
        "ORDER BY resolved_at DESC LIMIT ?",
        (category, extreme_side, int(window)),
    ))
    n = len(rows)
    wins = sum(1 for (r,) in rows if int(r) == 1)
    return n, wins


def load_category_state(
    conn: sqlite3.Connection,
    *,
    category: str,
    extreme_side: str = LONGSHOT_SIDE,
) -> CategoryState | None:
    """Load the persisted stop-loss timestamp for a category, if any."""
    row = conn.execute(
        "SELECT n_resolved_last20, n_wins_last20, win_rate_last20, stop_loss_until_ts "
        "FROM category_priors WHERE category=? AND extreme_side=?",
        (category, extreme_side),
    ).fetchone()
    if row is None:
        return None
    n_resolved, n_wins, win_rate, until_iso = row
    return CategoryState(
        category=category,
        extreme_side=extreme_side,
        n_resolved_last20=int(n_resolved or 0),
        n_wins_last20=int(n_wins or 0),
        win_rate_last20=float(win_rate) if win_rate is not None else None,
        stop_loss_until=_parse_dt(until_iso),
    )


def record_outcome(
    conn: sqlite3.Connection,
    *,
    category: str,
    extreme_side: str,
    market_id: str,
    win: bool,
    resolved_at: datetime | None = None,
    layer: LongshotLayer | None = None,
    now: datetime | None = None,
) -> CategoryState:
    """Append a resolved outcome and refresh the rolling win-rate + stop-loss state."""
    layer = layer or LongshotLayer()
    now_dt = now or datetime.now(tz=UTC)
    resolved_iso = (resolved_at or now_dt).astimezone(UTC).isoformat()
    conn.execute(
        "INSERT INTO longshot_outcomes(category, extreme_side, market_id, result, resolved_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (category, extreme_side, market_id, 1 if win else 0, resolved_iso),
    )
    n, wins = _read_window(
        conn, category=category, extreme_side=extreme_side, window=layer.stop_loss_window,
    )
    win_rate = (wins / n) if n else None
    stop_until: str | None = None
    if (
        win_rate is not None
        and n >= layer.stop_loss_window
        and win_rate < layer.stop_loss_win_rate_threshold
    ):
        stop_until = (now_dt + timedelta(hours=layer.stop_loss_suspension_hours)).astimezone(UTC).isoformat()
    alpha_existing = conn.execute(
        "SELECT alpha_adjustment FROM category_priors WHERE category=? AND extreme_side=?",
        (category, extreme_side),
    ).fetchone()
    alpha_value = float(alpha_existing[0]) if alpha_existing is not None else layer.alpha_for(category)
    conn.execute(
        "INSERT OR REPLACE INTO category_priors"
        "(category, extreme_side, alpha_adjustment, n_resolved_last20, n_wins_last20, "
        "win_rate_last20, stop_loss_until_ts, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            category, extreme_side, alpha_value, n, wins,
            win_rate, stop_until, now_dt.isoformat(),
        ),
    )
    return CategoryState(
        category=category,
        extreme_side=extreme_side,
        n_resolved_last20=n,
        n_wins_last20=wins,
        win_rate_last20=win_rate,
        stop_loss_until=_parse_dt(stop_until),
    )


# --- per-tick longshot evaluation -----------------------------------------


@dataclass(frozen=True)
class LongshotShadowRow:
    market_id: str
    category: str
    extreme_side: str
    p_market: float | None
    alpha: float
    p_prior: float | None
    decision: str
    reject_reason: str | None
    audit_json: str

    def to_row(
        self,
        *,
        ctx: TickContext,
        edge_source: str = "longshot_prior",
        sigma_p: float = 0.12,
    ) -> dict[str, Any]:
        return {
            "variant_name": "longshot",
            "tick_ts": ctx.tick_id,
            "experiment_id": ctx.experiment_id,
            "participant_idx": ctx.participant_idx,
            "market_id": self.market_id,
            "edge_source": edge_source,
            "p_mean": self.p_prior,
            "sigma_p": sigma_p,
            "side": self.extreme_side,
            "score": None,
            "decision": self.decision,
            "reject_reason": self.reject_reason,
            "audit_json": self.audit_json,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }


def _count_open_longshots(portfolio: PortfolioView, longshot_market_ids: set[str]) -> int:
    """Approximate the count of currently held longshot positions.

    We don't have a persisted "thesis" tag yet (Phase 1 doesn't track
    longshot vs LLM entries), so we count any open position on a market
    that *currently* meets the longshot/favorite criteria. That's
    conservative -- it overcounts when a market drifts in or out of the
    extreme bands.
    """
    return sum(
        1 for p in portfolio.positions if p.shares > 0 and p.market_id in longshot_market_ids
    )


def evaluate_market(
    market: MarketView,
    *,
    ctx: TickContext,
    portfolio: PortfolioView,
    layer: LongshotLayer,
    conn: sqlite3.Connection | None,
    open_longshot_count: int,
) -> LongshotShadowRow | None:
    """Build a :class:`LongshotShadowRow` for one market, or None if not extreme."""
    extreme_side = classify_extreme_side(market)
    if extreme_side is None:
        return None
    category = (market.topic or "other").lower().strip()
    q = market.quote
    p_market = compute_market_prior(best_bid=q.best_bid, best_ask=q.best_ask)
    alpha = layer.alpha_for(category)
    if p_market is None:
        audit = {
            "category": category,
            "extreme_side": extreme_side,
            "alpha": alpha,
            "reject_reason": REJECT_NO_MARKET_PRIOR,
        }
        return LongshotShadowRow(
            market_id=market.market_id, category=category, extreme_side=extreme_side,
            p_market=None, alpha=alpha, p_prior=None,
            decision=DECISION_BLOCKED, reject_reason=REJECT_NO_MARKET_PRIOR,
            audit_json=json.dumps(audit, default=str),
        )
    p_prior = _clip(p_market * alpha, P_MIN, P_MAX)

    reject: str | None = None
    horizon_hours = market.hours_to_resolution(ctx.now)
    horizon_days = (horizon_hours or 0.0) / 24.0
    if horizon_hours is None or horizon_days > layer.max_time_to_resolve_days:
        reject = REJECT_TIME_HORIZON
    elif open_longshot_count >= layer.max_open_longshots:
        reject = REJECT_OPEN_CAP
    elif conn is not None:
        state = load_category_state(conn, category=category, extreme_side=extreme_side)
        if state is not None and state.in_stop_loss(now=ctx.now):
            reject = REJECT_STOP_LOSS
    decision = DECISION_BLOCKED if reject else DECISION_ELIGIBLE
    audit = {
        "category": category,
        "extreme_side": extreme_side,
        "alpha": alpha,
        "p_market": p_market,
        "p_prior": p_prior,
        "horizon_days": horizon_days,
        "open_longshot_count": open_longshot_count,
        "decision": decision,
        "reject_reason": reject,
    }
    return LongshotShadowRow(
        market_id=market.market_id, category=category, extreme_side=extreme_side,
        p_market=p_market, alpha=alpha, p_prior=p_prior,
        decision=decision, reject_reason=reject,
        audit_json=json.dumps(audit, default=str),
    )


@dataclass(frozen=True)
class LongshotPassResult:
    shadow_rows: list[LongshotShadowRow] = field(default_factory=list)
    eligible_count: int = 0
    blocked_count: int = 0

    def to_shadow_rows(self, *, ctx: TickContext) -> list[dict[str, Any]]:
        return [r.to_row(ctx=ctx) for r in self.shadow_rows]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible_count,
            "blocked": self.blocked_count,
            "total": len(self.shadow_rows),
        }


def run_longshot_pass(
    *,
    ctx: TickContext,
    portfolio: PortfolioView,
    candidates: Iterable[MarketView],
    layer: LongshotLayer | None = None,
    conn: sqlite3.Connection | None = None,
) -> LongshotPassResult:
    """Scan every loaded candidate for longshot/favorite eligibility."""
    layer = layer or LongshotLayer()
    if not layer.enabled:
        return LongshotPassResult()
    candidates = list(candidates)
    extreme_ids: set[str] = set()
    for market in candidates:
        if classify_extreme_side(market) is not None:
            extreme_ids.add(market.market_id)
    open_longshots = _count_open_longshots(portfolio, extreme_ids)
    rows: list[LongshotShadowRow] = []
    eligible = 0
    blocked = 0
    for market in candidates:
        row = evaluate_market(
            market,
            ctx=ctx,
            portfolio=portfolio,
            layer=layer,
            conn=conn,
            open_longshot_count=open_longshots,
        )
        if row is None:
            continue
        rows.append(row)
        if row.decision == DECISION_ELIGIBLE:
            eligible += 1
        else:
            blocked += 1
    return LongshotPassResult(
        shadow_rows=rows, eligible_count=eligible, blocked_count=blocked,
    )
