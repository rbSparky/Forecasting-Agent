"""Hard-cap allocator and exit decision helpers.

The allocator turns scored sizing proposals into a final list of
:class:`ai_prophet_core.TradeIntentRequest` objects after enforcing every
server- and team-side cap.

Caps enforced (server-side from
``ai_prophet_core.ruleset`` plus internal additions):

- Max 30 open positions.
- Max $1,000 notional per ``(market_id, side)``.
- Max $10,000 gross exposure.
- Max 20 trades per tick.
- Max 60 trades per day (internal, well below the 100/day server cap).
- Min trade size $25 (sized at sizing layer, re-enforced here).
- Refuses to open a side opposite an already-held position on the same
  market.
- Refuses entry intents on a market where we already have an open
  position with a different proposed side (no flipping in a single
  intent).

Exit decisions (A.10) are exposed as pure functions: hard, soft, no-exit
zones. Phase 2 keeps soft exits *advisory only* and does not aggressively
generate exit intents -- the strategy layer chooses to act on them or
not. The deterministic dry-run strategy ignores soft exits entirely.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ai_prophet_core import TradeIntentRequest
from ai_prophet_core.ruleset import (
    MAX_GROSS_EXPOSURE,
    MAX_NOTIONAL_PER_MARKET,
    MAX_OPEN_POSITIONS,
    MAX_TRADES_PER_DAY,
    MAX_TRADES_PER_TICK,
)

from kalibre.score import FILL_PROB_SKIP_THRESHOLD, TRADE_SCORE_GATE
from kalibre.tick_context import PortfolioView, PositionView


# Internal daily trade cap (60) sits below the 100/day server cap so we
# never hit it unintentionally.
INTERNAL_DAILY_TRADE_CAP = 60


# --- proposal type ----------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """Scored entry candidate ready for the allocator."""

    market_id: str
    action: str  # 'BUY'
    side: str  # 'YES' | 'NO'
    size_usd: float
    shares: float
    side_price: float
    p_mean: float
    p_eff: float
    sigma_p: float
    score: float
    fill_prob: float
    edge_source: str = "external"
    topic: str | None = None
    source: str | None = None
    decision: str = "propose"
    reject_reason: str | None = None
    # Phase 6R: per-proposal extras (fill_prob component decomposition,
    # canary-size simulations, quote_age_sec, etc.). The dict is dumped
    # verbatim into the proposal audit_json; never used for risk gating.
    audit_extra: dict[str, Any] = field(default_factory=dict)


# --- allocator --------------------------------------------------------------


@dataclass(frozen=True)
class AllocatorConfig:
    max_open_positions: int = MAX_OPEN_POSITIONS
    max_notional_per_market_usd: float = MAX_NOTIONAL_PER_MARKET
    max_gross_exposure_usd: float = MAX_GROSS_EXPOSURE
    max_trades_per_tick: int = MAX_TRADES_PER_TICK
    server_daily_trade_cap: int = MAX_TRADES_PER_DAY
    internal_daily_trade_cap: int = INTERNAL_DAILY_TRADE_CAP
    score_gate: float = TRADE_SCORE_GATE
    fill_prob_skip_threshold: float = FILL_PROB_SKIP_THRESHOLD
    min_trade_usd: float = 25.0


REJECT_SCORE = "score_below_gate"
REJECT_FILL_PROB = "fill_prob_below_skip"
REJECT_DUPLICATE_OPPOSITE = "duplicate_opposite_side"
REJECT_DUPLICATE_OPEN = "already_open_same_side"
REJECT_PER_MARKET_CAP = "per_market_cap"
REJECT_GROSS_CAP = "gross_exposure_cap"
REJECT_OPEN_POSITIONS_CAP = "open_positions_cap"
REJECT_TICK_TRADE_CAP = "tick_trade_cap"
REJECT_DAILY_TRADE_CAP_INTERNAL = "internal_daily_trade_cap"
REJECT_DAILY_TRADE_CAP_SERVER = "server_daily_trade_cap"
REJECT_BELOW_MIN_TRADE = "size_below_min_after_clip"
REJECT_INVALID = "invalid_proposal"


@dataclass
class AllocationResult:
    intents: list[TradeIntentRequest]
    accepted: list[Proposal]
    rejected: list[Proposal]
    final_size_by_market: dict[tuple[str, str], float] = field(default_factory=dict)
    open_position_count_after: int = 0
    gross_exposure_after: float = 0.0
    trades_this_tick: int = 0


def _initial_state(portfolio: PortfolioView) -> dict[str, Any]:
    open_positions = sum(1 for p in portfolio.positions if p.shares > 0)
    held_side = portfolio.side_by_market()
    exposure_by_market = portfolio.exposure_by_market()
    gross = portfolio.gross_exposure_usd
    return {
        "open_positions": open_positions,
        "held_side": held_side,
        "exposure_by_market": dict(exposure_by_market),
        "gross_exposure": gross,
    }


def _record_reject(proposals: list[Proposal], proposal: Proposal, reason: str) -> Proposal:
    rejected = Proposal(
        market_id=proposal.market_id,
        action=proposal.action,
        side=proposal.side,
        size_usd=proposal.size_usd,
        shares=proposal.shares,
        side_price=proposal.side_price,
        p_mean=proposal.p_mean,
        p_eff=proposal.p_eff,
        sigma_p=proposal.sigma_p,
        score=proposal.score,
        fill_prob=proposal.fill_prob,
        edge_source=proposal.edge_source,
        topic=proposal.topic,
        source=proposal.source,
        decision="reject",
        reject_reason=reason,
        audit_extra=dict(proposal.audit_extra),
    )
    proposals.append(rejected)
    return rejected


def allocate(
    proposals: Iterable[Proposal],
    *,
    portfolio: PortfolioView,
    daily_fills_already: int,
    config: AllocatorConfig = AllocatorConfig(),
) -> AllocationResult:
    """Enforce caps and produce intents in decreasing-score order.

    ``daily_fills_already`` is the count of fills that have already
    happened in the rolling 24h window; rejecting at or above
    ``server_daily_trade_cap`` and ``internal_daily_trade_cap`` keeps us
    safe even if the strategy proposed too many.
    """
    state = _initial_state(portfolio)
    accepted: list[Proposal] = []
    audit: list[Proposal] = []
    intents: list[TradeIntentRequest] = []
    trades_this_tick = 0

    # Score-ordered (descending) view for fair allocation.
    sorted_props = sorted(
        list(proposals), key=lambda p: (-p.score, p.market_id, p.side),
    )

    for proposal in sorted_props:
        if proposal.decision != "propose":
            audit.append(proposal)
            continue

        # Validate basic invariants first.
        if proposal.side not in ("YES", "NO") or proposal.action != "BUY":
            _record_reject(audit, proposal, REJECT_INVALID)
            continue
        if proposal.size_usd <= 0 or proposal.shares <= 0:
            _record_reject(audit, proposal, REJECT_INVALID)
            continue

        # Score / fill_prob gates.
        if proposal.score < config.score_gate:
            _record_reject(audit, proposal, REJECT_SCORE)
            continue
        if proposal.fill_prob < config.fill_prob_skip_threshold:
            _record_reject(audit, proposal, REJECT_FILL_PROB)
            continue

        # Duplicate-side checks.
        existing_side = state["held_side"].get(proposal.market_id)
        if existing_side is not None:
            if existing_side != proposal.side:
                _record_reject(audit, proposal, REJECT_DUPLICATE_OPPOSITE)
                continue
            # Same-side rebalance is allowed but not in Phase 2 baseline.
            _record_reject(audit, proposal, REJECT_DUPLICATE_OPEN)
            continue

        # Per-tick trade cap.
        if trades_this_tick >= config.max_trades_per_tick:
            _record_reject(audit, proposal, REJECT_TICK_TRADE_CAP)
            continue
        # Daily trade caps. Internal first (the tighter one).
        projected_today = daily_fills_already + trades_this_tick
        if projected_today >= config.internal_daily_trade_cap:
            _record_reject(audit, proposal, REJECT_DAILY_TRADE_CAP_INTERNAL)
            continue
        if projected_today >= config.server_daily_trade_cap:
            _record_reject(audit, proposal, REJECT_DAILY_TRADE_CAP_SERVER)
            continue

        # Open-position cap.
        if state["open_positions"] >= config.max_open_positions:
            _record_reject(audit, proposal, REJECT_OPEN_POSITIONS_CAP)
            continue

        # Per-market notional cap.
        per_market_used = state["exposure_by_market"].get(proposal.market_id, 0.0)
        per_market_remaining = config.max_notional_per_market_usd - per_market_used
        if per_market_remaining <= 0:
            _record_reject(audit, proposal, REJECT_PER_MARKET_CAP)
            continue

        # Gross exposure cap.
        gross_remaining = config.max_gross_exposure_usd - state["gross_exposure"]
        if gross_remaining <= 0:
            _record_reject(audit, proposal, REJECT_GROSS_CAP)
            continue

        cap = min(proposal.size_usd, per_market_remaining, gross_remaining)
        if cap < config.min_trade_usd:
            _record_reject(audit, proposal, REJECT_BELOW_MIN_TRADE)
            continue

        final_size_usd = cap
        final_shares = (
            proposal.shares
            if math.isclose(final_size_usd, proposal.size_usd)
            else final_size_usd / max(1e-9, proposal.side_price)
        )

        intent = TradeIntentRequest(
            market_id=proposal.market_id,
            action=proposal.action,
            side=proposal.side,
            shares=_format_shares(final_shares),
            idempotency_key="",
        )
        intents.append(intent)

        # Update state for downstream proposals.
        state["open_positions"] += 1
        state["exposure_by_market"][proposal.market_id] = per_market_used + final_size_usd
        state["gross_exposure"] += final_size_usd
        state["held_side"][proposal.market_id] = proposal.side
        trades_this_tick += 1

        accepted_proposal = Proposal(
            market_id=proposal.market_id,
            action=proposal.action,
            side=proposal.side,
            size_usd=final_size_usd,
            shares=final_shares,
            side_price=proposal.side_price,
            p_mean=proposal.p_mean,
            p_eff=proposal.p_eff,
            sigma_p=proposal.sigma_p,
            score=proposal.score,
            fill_prob=proposal.fill_prob,
            edge_source=proposal.edge_source,
            topic=proposal.topic,
            source=proposal.source,
            decision="accept",
            reject_reason=None,
            audit_extra=dict(proposal.audit_extra),
        )
        accepted.append(accepted_proposal)
        audit.append(accepted_proposal)

    return AllocationResult(
        intents=intents,
        accepted=accepted,
        rejected=[p for p in audit if p.decision == "reject"],
        final_size_by_market={
            (p.market_id, p.side): p.size_usd for p in accepted
        },
        open_position_count_after=state["open_positions"],
        gross_exposure_after=state["gross_exposure"],
        trades_this_tick=trades_this_tick,
    )


def _format_shares(n: float) -> str:
    """Format ``n`` as a string with up to 4 decimal places (server-friendly)."""
    if n <= 0:
        return "0"
    rounded = round(n, 4)
    if rounded <= 0:
        return "0"
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return text or "0"


# --- A.10 exit decisions ----------------------------------------------------


@dataclass(frozen=True)
class ExitContext:
    position: PositionView
    current_bid: float | None
    current_ask: float | None
    spread: float | None
    entry_cost_usd: float | None
    time_to_resolution_hours: float | None
    p_market_now: float | None
    p_cal_at_entry: float | None
    market_invalidated: bool = False
    violates_cap: bool = False
    position_size_usd: float = 0.0


HARD_EXIT_MARKET_INVALIDATED = "hard_exit_market_invalidated"
HARD_EXIT_CAP_VIOLATION = "hard_exit_cap_violation"
HARD_EXIT_SEVERE_DRAWDOWN = "hard_exit_severe_drawdown"
HARD_EXIT_PRE_RESOLUTION_DRIFT = "hard_exit_pre_resolution_drift"

NO_EXIT_ZONE_WIDE_SPREAD = "no_exit_wide_spread"
NO_EXIT_ZONE_TINY_POSITION = "no_exit_tiny_position"

SOFT_EXIT_EDGE_DECAY = "soft_exit_edge_decay"
SOFT_EXIT_SLOT_PRESSURE = "soft_exit_slot_pressure"


@dataclass(frozen=True)
class ExitDecision:
    market_id: str
    side: str
    decision: str  # 'hold' | 'hard_exit' | 'soft_exit_advisory'
    reason: str


def evaluate_hard_exit(ctx: ExitContext) -> ExitDecision | None:
    pos = ctx.position
    if ctx.market_invalidated:
        return ExitDecision(pos.market_id, pos.side, "hard_exit", HARD_EXIT_MARKET_INVALIDATED)
    if ctx.violates_cap:
        return ExitDecision(pos.market_id, pos.side, "hard_exit", HARD_EXIT_CAP_VIOLATION)
    if ctx.entry_cost_usd is not None and pos.unrealized_pnl < -0.40 * abs(ctx.entry_cost_usd):
        return ExitDecision(pos.market_id, pos.side, "hard_exit", HARD_EXIT_SEVERE_DRAWDOWN)
    if (
        ctx.time_to_resolution_hours is not None
        and ctx.time_to_resolution_hours < 1.0
        and ctx.p_market_now is not None
        and ctx.p_cal_at_entry is not None
        and abs(ctx.p_market_now - ctx.p_cal_at_entry) > 0.30
    ):
        return ExitDecision(pos.market_id, pos.side, "hard_exit", HARD_EXIT_PRE_RESOLUTION_DRIFT)
    return None


def in_no_exit_zone(ctx: ExitContext) -> tuple[bool, str | None]:
    if ctx.spread is not None and ctx.spread > 0.06:
        return True, NO_EXIT_ZONE_WIDE_SPREAD
    if ctx.position_size_usd > 0 and ctx.position_size_usd < 30.0:
        return True, NO_EXIT_ZONE_TINY_POSITION
    return False, None


def hold_score(
    *,
    shares: float,
    p_cal: float,
    current_price_for_side: float,
    expected_exit_spread_cost_usd: float,
    capital_days_remaining: float,
) -> float:
    profit = shares * (p_cal - current_price_for_side) - max(0.0, expected_exit_spread_cost_usd)
    denom = max(1e-9, capital_days_remaining)
    return profit / denom


def exit_score(
    *,
    released_capital_usd: float,
    best_alternative_score: float,
    exit_cost_usd: float,
    exit_slippage_usd: float,
) -> float:
    return (
        max(0.0, released_capital_usd) * best_alternative_score
        - max(0.0, exit_cost_usd)
        - max(0.0, exit_slippage_usd)
    )


def should_soft_exit(
    *,
    hold_score_value: float,
    exit_score_value: float,
    buffer: float = 0.01,
) -> bool:
    return exit_score_value > hold_score_value + buffer
