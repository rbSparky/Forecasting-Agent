"""Phase 2 deterministic strategy engine.

The engine consumes a :class:`TickContext` plus an explicit
``probabilities_by_market`` map and emits proposals + intents. It is
deliberately *non-alpha*: nothing here generates probabilities. If no
probability is supplied for a market, the engine returns HOLD. Tests and
later phases inject probabilities (eventually from the LLM forecaster).

Live trades are gated twice:

- The engine itself only runs in ``deterministic_dry_run`` mode.
- The loop refuses to actually submit non-empty intents unless the
  caller-side flag ``enable_live_trades`` is True (the loop reads this
  from ``KALIBRE_ENABLE_LIVE_TRADES``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ai_prophet_core import TradeIntentRequest

from kalibre.portfolio import (
    AllocationResult,
    AllocatorConfig,
    Proposal,
    allocate,
)
from kalibre.score import (
    DAILY_HARD_CAP_DEFAULT_USD,
    FILL_PROB_SKIP_THRESHOLD,
    TRADE_SCORE_GATE,
    canary_fill_prob,
    evaluate_proposal_score,
    fill_prob_components,
)
from kalibre.sizing import (
    SizingInputs,
    SizingResult,
    size_proposal,
)
from kalibre.tick_context import MarketView, TickContext
from kalibre.universe import (
    UniverseDecision,
    UniverseFilterConfig,
    filter_universe,
)


# --- public types -----------------------------------------------------------


class StrategyMode(str, Enum):
    T0_NOOP = "t0_noop"
    DETERMINISTIC_DRY_RUN = "deterministic_dry_run"
    FORECAST_DRY_RUN = "forecast_dry_run"


@dataclass(frozen=True)
class MarketProbability:
    """Probability input for one market.

    ``p_mean`` and ``sigma_p`` are the *post-market-blend* values the
    strategy consumes. The optional ``p_model_only`` / ``sigma_model_only``
    fields preserve the *pre-market-blend* calibrated model probability
    so a later combiner (Phase 6R Opus path) can combine raw model
    estimates without double-counting the market prior.
    """

    p_mean: float
    sigma_p: float = 0.10
    edge_source: str = "external"
    model_tier: str = "external"
    p_model_only: float | None = None
    sigma_model_only: float | None = None


@dataclass
class StrategyResult:
    mode: str
    proposals: list[Proposal]
    intents: list[TradeIntentRequest]
    universe_decisions: tuple[UniverseDecision, ...]
    audit_rows: list[dict[str, Any]]
    allocation: AllocationResult | None = None
    live_trades_enabled: bool = False
    notes: list[str] = field(default_factory=list)
    missing_probability_count: int = 0

    @property
    def proposal_summary(self) -> dict[str, Any]:
        accepted = [p for p in self.proposals if p.decision == "accept"]
        rejected = [p for p in self.proposals if p.decision == "reject"]
        reject_counts = _count_by(rejected, lambda p: p.reject_reason or "unknown")
        if self.missing_probability_count:
            # Surface missing-probability holds in the same reject_reason
            # histogram so the put_plan summary is decision-coverage-complete.
            reject_counts["no_probability"] = reject_counts.get(
                "no_probability", 0,
            ) + self.missing_probability_count
        return {
            "mode": self.mode,
            "live_trades_enabled": self.live_trades_enabled,
            "universe_candidates": sum(1 for d in self.universe_decisions if d.accepted),
            "universe_rejected": sum(1 for d in self.universe_decisions if not d.accepted),
            "missing_probability": self.missing_probability_count,
            "proposals_accepted": len(accepted),
            "proposals_rejected": len(rejected) + self.missing_probability_count,
            "intents_built": len(self.intents),
            "reject_reason_counts": reject_counts,
            "universe_reject_reason_counts": _count_by(
                [d for d in self.universe_decisions if not d.accepted],
                lambda d: d.reject_reason or "unknown",
            ),
        }


def _count_by(items: list[Any], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = key(it)
        out[k] = out.get(k, 0) + 1
    return out


# --- strategy entrypoints ---------------------------------------------------


def t0_noop_strategy(ctx: TickContext) -> StrategyResult:
    """Phase-1 behavior. Always returns an empty intent list."""
    return StrategyResult(
        mode=StrategyMode.T0_NOOP.value,
        proposals=[],
        intents=[],
        universe_decisions=(),
        audit_rows=[],
        allocation=None,
        live_trades_enabled=False,
        notes=["t0_noop"],
    )


def deterministic_dry_run_strategy(
    ctx: TickContext,
    *,
    probabilities: Mapping[str, MarketProbability],
    enable_live_trades: bool = False,
    universe_config: UniverseFilterConfig = UniverseFilterConfig(),
    allocator_config: AllocatorConfig = AllocatorConfig(),
    n_resolved: int = 0,
    spent_today_usd: float = 0.0,
    daily_fills_already: int = 0,
    score_gate: float = TRADE_SCORE_GATE,
    fill_prob_skip_threshold: float = FILL_PROB_SKIP_THRESHOLD,
    daily_api_hard_cap_usd: float = DAILY_HARD_CAP_DEFAULT_USD,
) -> StrategyResult:
    """Run universe -> sizing -> scoring -> allocator on the supplied context.

    ``daily_api_hard_cap_usd`` feeds the A.5.3 API-budget shadow price. It
    is the *spend-governor's* daily hard cap (default $35), NOT the gross
    exposure cap. Passing the wrong cap here makes the scarcity term
    effectively zero on liquid days and starves trades on illiquid ones.
    """
    accepted, decisions = filter_universe(ctx.markets, now=ctx.now, config=universe_config)
    market_by_id = ctx.market_by_id()
    exposure_by_market = ctx.portfolio.exposure_by_market()
    exposure_by_topic = ctx.portfolio.exposure_by_topic(market_by_id)
    slots_used = ctx.portfolio.open_position_count
    remaining_gross_cap = max(
        0.0,
        allocator_config.max_gross_exposure_usd - ctx.portfolio.gross_exposure_usd,
    )
    bankroll = max(0.0, ctx.portfolio.equity or 0.0)

    proposals: list[Proposal] = []
    missing_probability: list[MarketView] = []
    sizing_results: dict[str, SizingResult] = {}

    for market in accepted:
        prob = probabilities.get(market.market_id)
        if prob is None:
            missing_probability.append(market)
            continue
        sizing = _size_one(
            market=market,
            prob=prob,
            ctx=ctx,
            bankroll=bankroll,
            exposure_by_market=exposure_by_market,
            exposure_by_topic=exposure_by_topic,
            remaining_gross_cap=remaining_gross_cap,
            n_resolved=n_resolved,
            allocator_config=allocator_config,
        )
        sizing_results[market.market_id] = sizing
        if sizing.action == "HOLD":
            proposals.append(_proposal_from_hold(market, sizing, prob))
            continue
        prop = _score_proposal(
            market=market,
            sizing=sizing,
            prob=prob,
            ctx=ctx,
            slots_used=slots_used,
            topic_exposure_usd=exposure_by_topic.get(market.category, 0.0),
            spent_today_usd=spent_today_usd,
            daily_hard_cap_usd=daily_api_hard_cap_usd,
            score_gate=score_gate,
            fill_prob_skip_threshold=fill_prob_skip_threshold,
        )
        proposals.append(prop)

    allocation = allocate(
        [p for p in proposals if p.decision == "propose"],
        portfolio=ctx.portfolio,
        daily_fills_already=daily_fills_already,
        config=allocator_config,
    )

    # Merge allocator audit (accept/reject) on top of pre-allocator HOLD rows.
    final_proposals = (
        [p for p in proposals if p.decision != "propose"]
        + allocation.accepted
        + allocation.rejected
    )

    intents = allocation.intents if enable_live_trades else []
    audit_rows = [
        _audit_row(p, ctx) for p in final_proposals
    ] + [
        _missing_probability_audit_row(m, ctx) for m in missing_probability
    ] + [
        _universe_reject_audit_row(d, ctx) for d in decisions if not d.accepted
    ]

    notes = []
    if not enable_live_trades and allocation.intents:
        notes.append("live_trades_disabled: intents withheld")
    if missing_probability:
        notes.append(f"missing_probability: {len(missing_probability)} candidate(s) held")

    return StrategyResult(
        mode=StrategyMode.DETERMINISTIC_DRY_RUN.value,
        proposals=final_proposals,
        intents=list(intents),
        universe_decisions=decisions,
        audit_rows=audit_rows,
        allocation=allocation,
        live_trades_enabled=enable_live_trades,
        notes=notes,
        missing_probability_count=len(missing_probability),
    )


# --- helpers ---------------------------------------------------------------


def _size_one(
    *,
    market: MarketView,
    prob: MarketProbability,
    ctx: TickContext,
    bankroll: float,
    exposure_by_market: dict[str, float],
    exposure_by_topic: dict[str, float],
    remaining_gross_cap: float,
    n_resolved: int,
    allocator_config: AllocatorConfig,
) -> SizingResult:
    inputs = SizingInputs(
        market_id=market.market_id,
        best_bid=market.quote.best_bid or 0.0,
        best_ask=market.quote.best_ask or 0.0,
        p_mean=prob.p_mean,
        sigma_p=prob.sigma_p,
        bankroll_usd=bankroll,
        exposure_market_usd=exposure_by_market.get(market.market_id, 0.0),
        topic_exposure_usd=exposure_by_topic.get(market.category, 0.0),
        pnl_24h=ctx.pnl_24h,
        remaining_gross_cap_usd=remaining_gross_cap,
        n_resolved=n_resolved,
        server_per_market_cap_usd=allocator_config.max_notional_per_market_usd,
    )
    return size_proposal(inputs)


def _score_proposal(
    *,
    market: MarketView,
    sizing: SizingResult,
    prob: MarketProbability,
    ctx: TickContext,
    slots_used: int,
    topic_exposure_usd: float,
    spent_today_usd: float,
    daily_hard_cap_usd: float,
    score_gate: float,
    fill_prob_skip_threshold: float,
) -> Proposal:
    bid = market.quote.best_bid or 0.0
    ask = market.quote.best_ask or 0.0
    spread = market.quote.spread or 0.0
    quote_age_sec = market.quote.quote_age_sec(ctx.now) or 0.0
    side_price = ask if sizing.side == "YES" else (1.0 - bid)
    time_to_resolve_days = max(
        0.1, (market.hours_to_resolution(ctx.now) or 24.0) / 24.0,
    )
    breakdown = evaluate_proposal_score(
        side=sizing.side or "YES",
        p_eff=sizing.p_eff,
        bid=bid,
        ask=ask,
        shares=sizing.shares,
        size_usd=sizing.size_usd,
        spread=spread,
        time_to_resolve_days=time_to_resolve_days,
        slots_used=slots_used,
        topic_exposure_usd=topic_exposure_usd,
        spent_today_usd=spent_today_usd,
        daily_hard_cap_usd=daily_hard_cap_usd,
        source=market.source,
        volume_24h=market.quote.volume_24h,
        quote_age_sec=quote_age_sec,
    )

    decision = "propose"
    reject_reason: str | None = None
    # Phase 6: Opus escalation flags markets where Sonnet and Opus disagree
    # by more than the configured threshold. We hold rather than trade --
    # auditable, no surprise short-side exposure.
    if prob.edge_source == "model_disagreement":
        decision, reject_reason = "reject", "model_disagreement"
    elif breakdown.score < score_gate:
        decision, reject_reason = "reject", "score_below_gate"
    elif breakdown.fill_prob < fill_prob_skip_threshold:
        decision, reject_reason = "reject", "fill_prob_below_skip"

    # Phase 6R: per-proposal audit additions (fill-prob component
    # decomposition + canary-size simulations + raw quote age). Used by
    # the canary fill-rescue shadow stream and the funnel report; never
    # used to mutate the gating decision above.
    components = fill_prob_components(
        source=market.source,
        volume_24h=market.quote.volume_24h,
        spread=spread,
        shares=sizing.shares,
        quote_age_sec=quote_age_sec,
    )
    _, canary_fp_25 = canary_fill_prob(
        source=market.source,
        volume_24h=market.quote.volume_24h,
        spread=spread,
        side_price=side_price,
        quote_age_sec=quote_age_sec,
        canary_size_usd=25.0,
    )
    _, canary_fp_50 = canary_fill_prob(
        source=market.source,
        volume_24h=market.quote.volume_24h,
        spread=spread,
        side_price=side_price,
        quote_age_sec=quote_age_sec,
        canary_size_usd=50.0,
    )
    audit_extra = {
        "original_size_usd": float(sizing.size_usd),
        "original_shares": float(sizing.shares),
        "original_fill_prob": float(breakdown.fill_prob),
        "canary_size_usd_25": 25.0,
        "canary_fill_prob_25": float(canary_fp_25),
        "canary_size_usd_50": 50.0,
        "canary_fill_prob_50": float(canary_fp_50),
        "quote_age_sec": float(quote_age_sec),
        "fill_prob_block_components": components.to_dict(),
        "side_price": float(side_price),
    }

    return Proposal(
        market_id=market.market_id,
        action="BUY",
        side=sizing.side or "YES",
        size_usd=sizing.size_usd,
        shares=sizing.shares,
        side_price=side_price,
        p_mean=prob.p_mean,
        p_eff=sizing.p_eff,
        sigma_p=sizing.sigma_p,
        score=breakdown.score,
        fill_prob=breakdown.fill_prob,
        edge_source=prob.edge_source,
        topic=market.category,
        source=market.source,
        decision=decision,
        reject_reason=reject_reason,
        audit_extra=audit_extra,
    )


def _proposal_from_hold(
    market: MarketView,
    sizing: SizingResult,
    prob: MarketProbability,
) -> Proposal:
    side = sizing.side or "YES"
    side_price = (market.quote.best_ask or 0.0) if side == "YES" else (1.0 - (market.quote.best_bid or 0.0))
    return Proposal(
        market_id=market.market_id,
        action="BUY",
        side=side,
        size_usd=sizing.size_usd,
        shares=sizing.shares,
        side_price=side_price,
        p_mean=prob.p_mean,
        p_eff=sizing.p_eff,
        sigma_p=sizing.sigma_p,
        score=0.0,
        fill_prob=0.0,
        edge_source=prob.edge_source,
        topic=market.category,
        source=market.source,
        decision="reject",
        reject_reason=sizing.reject_reason or "sizing_hold",
    )


def _audit_row(p: Proposal, ctx: TickContext) -> dict[str, Any]:
    audit_payload: dict[str, Any] = {
        "edge_source": p.edge_source,
        "tick_ts": ctx.tick_id,
        "version": ctx.version,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": p.market_id,
        "topic": p.topic,
        "source": p.source,
        "p_mean": p.p_mean,
        "p_eff": p.p_eff,
        "sigma_p": p.sigma_p,
        "score": p.score,
        "fill_prob": p.fill_prob,
        "decision": p.decision,
        "reject_reason": p.reject_reason,
    }
    # Phase 6R: merge per-proposal audit extras (fill_prob components,
    # canary-size simulations, quote_age_sec). Never overrides primary keys.
    if p.audit_extra:
        for k, v in p.audit_extra.items():
            audit_payload.setdefault(k, v)
    return {
        "proposal_id": f"{ctx.tick_id}:{p.market_id}:{p.side}",
        "tick_ts": ctx.tick_id,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": p.market_id,
        "edge_source": p.edge_source,
        "version": ctx.version,
        "p_mean": p.p_mean,
        "p_eff": p.p_eff,
        "sigma_p": p.sigma_p,
        "side": p.side,
        "action": p.action,
        "size_usd": p.size_usd,
        "shares": p.shares,
        "score": p.score,
        "fill_prob": p.fill_prob,
        "decision": p.decision,
        "reject_reason": p.reject_reason,
        "audit_json": json.dumps(audit_payload, default=str),
    }


def _missing_probability_audit_row(market: MarketView, ctx: TickContext) -> dict[str, Any]:
    """Audit row for a market that passed universe but has no probability.

    Every universe-accepted market either becomes a Proposal or gets this
    row, so ``len(universe_accepted) == accepted_proposals + rejected_proposals
    + missing_probability``. Decision coverage is preserved without invoking
    sizing/scoring on a market we can't actually evaluate.
    """
    return {
        "proposal_id": f"{ctx.tick_id}:{market.market_id}:no_probability",
        "tick_ts": ctx.tick_id,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": market.market_id,
        "edge_source": "missing_probability",
        "version": ctx.version,
        "p_mean": None,
        "p_eff": None,
        "sigma_p": None,
        "side": None,
        "action": None,
        "size_usd": None,
        "shares": None,
        "score": None,
        "fill_prob": None,
        "decision": "hold",
        "reject_reason": "no_probability",
        "audit_json": json.dumps(
            {
                "edge_source": "missing_probability",
                "tick_ts": ctx.tick_id,
                "version": ctx.version,
                "experiment_id": ctx.experiment_id,
                "participant_idx": ctx.participant_idx,
                "market_id": market.market_id,
                "topic": market.category,
                "source": market.source,
                "reject_reason": "no_probability",
                "decision": "hold",
                "best_bid": market.quote.best_bid,
                "best_ask": market.quote.best_ask,
                "volume_24h": market.quote.volume_24h,
                "quote_age_sec": market.quote.quote_age_sec(ctx.now),
                "hours_to_resolution": market.hours_to_resolution(ctx.now),
            },
            default=str,
        ),
    }


def _universe_reject_audit_row(d: UniverseDecision, ctx: TickContext) -> dict[str, Any]:
    return {
        "proposal_id": f"{ctx.tick_id}:{d.market_id}:universe",
        "tick_ts": ctx.tick_id,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": d.market_id,
        "edge_source": "universe_filter",
        "version": ctx.version,
        "p_mean": None,
        "p_eff": None,
        "sigma_p": None,
        "side": None,
        "action": None,
        "size_usd": None,
        "shares": None,
        "score": None,
        "fill_prob": None,
        "decision": "reject",
        "reject_reason": d.reject_reason,
        "audit_json": json.dumps(
            {
                "edge_source": "universe_filter",
                "tick_ts": ctx.tick_id,
                "version": ctx.version,
                "experiment_id": ctx.experiment_id,
                "participant_idx": ctx.participant_idx,
                "market_id": d.market_id,
                "spread": d.spread,
                "volume_24h": d.volume_24h,
                "quote_age_sec": d.quote_age_sec,
                "hours_to_resolution": d.hours_to_resolution,
                "source": d.source,
                "reject_reason": d.reject_reason,
            },
            default=str,
        ),
    }


# --- dispatcher used by the loop -------------------------------------------


@dataclass(frozen=True)
class StrategyEnvConfig:
    """Resolved-from-env configuration for :func:`dispatch_strategy`."""

    mode: StrategyMode
    enable_live_trades: bool


def parse_strategy_env(env: Mapping[str, str]) -> StrategyEnvConfig:
    raw_mode = (env.get("KALIBRE_STRATEGY_MODE") or "t0_noop").lower().strip()
    try:
        mode = StrategyMode(raw_mode)
    except ValueError:
        mode = StrategyMode.T0_NOOP
    enable_live = (env.get("KALIBRE_ENABLE_LIVE_TRADES") or "").strip() == "1"
    return StrategyEnvConfig(mode=mode, enable_live_trades=enable_live)
