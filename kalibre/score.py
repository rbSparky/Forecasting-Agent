"""Capital-efficiency score (AGENTS.md A.5) + fill model (A.9).

Pure functions. No I/O, no globals; defaults follow the AGENTS spec but
all knobs are explicit parameters so tests / future calibration can
override them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- thresholds from B and A.5/A.9 ------------------------------------------

TRADE_SCORE_GATE = 0.0005
FILL_PROB_SKIP_THRESHOLD = 0.35
PROB_EXIT_BEFORE_RESOLUTION_DEFAULT = 0.30
ALPHA_OPP_DEFAULT = 0.5
DAILY_HARD_CAP_DEFAULT_USD = 35.0

# Initial fill base probabilities, refit from observed fills after Day 2.
BASE_FILL_PROB = {
    "kalshi": 0.95,
    "polymarket": 0.92,
}
BASE_FILL_PROB_FALLBACK = 0.85


# --- A.5.1: expected profit -------------------------------------------------


def expected_value_per_share_yes(p_eff: float, ask: float) -> float:
    return p_eff - ask


def expected_value_per_share_no(p_eff: float, bid: float) -> float:
    return bid - p_eff


def expected_exit_spread_cost_usd(
    spread: float,
    shares: float,
    prob_exit_before_resolution: float = PROB_EXIT_BEFORE_RESOLUTION_DEFAULT,
) -> float:
    """``0.5 * spread * shares * prob_exit`` per A.5.1."""
    return 0.5 * max(0.0, spread) * max(0.0, shares) * max(0.0, prob_exit_before_resolution)


def expected_profit_usd(
    *,
    shares: float,
    expected_value_per_share: float,
    expected_exit_spread_cost: float = 0.0,
    expected_slippage_cost: float = 0.0,
    allocated_api_cost_usd: float = 0.0,
) -> float:
    """``shares * EV/share - exit_spread - slippage - api_cost``.

    Entry-spread cost is implicit in using ``ask`` / ``(1-bid)`` and is
    NOT subtracted again.
    """
    return (
        shares * expected_value_per_share
        - max(0.0, expected_exit_spread_cost)
        - max(0.0, expected_slippage_cost)
        - max(0.0, allocated_api_cost_usd)
    )


# --- A.5.2: capital-days ----------------------------------------------------


def capital_days(
    size_usd: float,
    time_to_resolve_days: float,
    expected_close_fraction: float = 1.0,
) -> float:
    held = max(0.1, max(0.0, expected_close_fraction) * max(0.0, time_to_resolve_days))
    return max(0.0, size_usd) * held


# --- A.5.3: scarcity shadow prices ------------------------------------------


def slot_shadow_price_usd(slots_used: int) -> float:
    if slots_used < 20:
        return 0.0
    if slots_used < 27:
        return 5.0 * ((slots_used - 20) ** 1.5)
    return 50.0 * ((slots_used - 27) ** 2) + 35.0


def risk_budget_shadow_price_usd(topic_exposure_usd: float, size_usd: float) -> float:
    over = max(0.0, max(0.0, topic_exposure_usd) + max(0.0, size_usd) - 1500.0)
    return over * 0.05


def api_budget_shadow_price_usd(
    spent_today_usd: float,
    daily_hard_cap_usd: float = DAILY_HARD_CAP_DEFAULT_USD,
) -> float:
    if daily_hard_cap_usd <= 0:
        return 0.0
    pct = max(0.0, spent_today_usd) / daily_hard_cap_usd
    if pct < 0.60:
        return 0.0
    if pct < 0.90:
        return 20.0 * (pct - 0.60)
    return 200.0 * (pct - 0.90) + 6.0


def total_scarcity_cost_usd(
    *,
    slots_used: int,
    topic_exposure_usd: float,
    size_usd: float,
    spent_today_usd: float,
    daily_hard_cap_usd: float = DAILY_HARD_CAP_DEFAULT_USD,
) -> float:
    return (
        slot_shadow_price_usd(slots_used)
        + risk_budget_shadow_price_usd(topic_exposure_usd, size_usd)
        + api_budget_shadow_price_usd(spent_today_usd, daily_hard_cap_usd)
    )


# --- A.5.4: final score -----------------------------------------------------


def final_score(
    expected_profit_usd: float,
    capital_days: float,
    scarcity_cost_usd: float,
) -> float:
    denom = max(0.0, capital_days) + max(0.0, scarcity_cost_usd)
    if denom <= 0.0:
        return 0.0
    return expected_profit_usd / denom


# --- A.9: fill probability heuristic ----------------------------------------


def liquidity_factor(volume_24h: float | None) -> float:
    if volume_24h is None:
        return 0.5
    return max(0.5, min(1.0, (volume_24h - 50.0) / 1950.0))


def spread_factor(spread: float | None) -> float:
    if spread is None:
        return 0.6
    return max(0.6, min(1.0, 1.0 - 4.0 * spread))


def aggressiveness_factor(shares: float, depth_est: float) -> float:
    if depth_est <= 0:
        return 0.5
    return 0.5 if shares > depth_est / 3.0 else 1.0


def recency_factor(quote_age_sec: float | None) -> float:
    if quote_age_sec is None:
        return 0.5
    if quote_age_sec <= 60:
        return 1.0
    if quote_age_sec >= 300:
        return 0.5
    # Linear decay from 1.0 at 60s to 0.5 at 300s.
    return 1.0 - 0.5 * (quote_age_sec - 60) / (300 - 60)


def fill_prob(
    *,
    source: str | None,
    volume_24h: float | None,
    spread: float | None,
    shares: float,
    quote_age_sec: float | None,
) -> float:
    """Compose the A.9 heuristic clipped into ``[0.30, 0.99]``."""
    base = BASE_FILL_PROB.get((source or "").lower(), BASE_FILL_PROB_FALLBACK)
    depth_est = ((volume_24h or 0.0) / 96.0)
    raw = (
        base
        * liquidity_factor(volume_24h)
        * spread_factor(spread)
        * aggressiveness_factor(shares, depth_est)
        * recency_factor(quote_age_sec)
    )
    return max(0.30, min(0.99, raw))


def fill_adjusted_size_usd(size_usd: float, fill_probability: float) -> float:
    """Dampen size by ``sqrt(fill_prob)``."""
    fp = max(0.0, min(1.0, fill_probability))
    return max(0.0, size_usd) * math.sqrt(fp)


def expected_profit_after_fill(
    *,
    fill_probability: float,
    expected_profit_if_filled: float,
    opportunity_cost_usd: float = 0.0,
) -> float:
    fp = max(0.0, min(1.0, fill_probability))
    return fp * expected_profit_if_filled - (1.0 - fp) * max(0.0, opportunity_cost_usd)


# --- decision helpers -------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    expected_value_per_share: float
    expected_profit_usd: float
    capital_days_value: float
    scarcity_cost_usd: float
    score_raw: float
    fill_prob: float
    score: float
    fill_adjusted_size_usd: float


def evaluate_proposal_score(
    *,
    side: str,
    p_eff: float,
    bid: float,
    ask: float,
    shares: float,
    size_usd: float,
    spread: float,
    time_to_resolve_days: float,
    slots_used: int,
    topic_exposure_usd: float,
    spent_today_usd: float = 0.0,
    daily_hard_cap_usd: float = DAILY_HARD_CAP_DEFAULT_USD,
    allocated_api_cost_usd: float = 0.0,
    expected_slippage_cost: float = 0.0,
    prob_exit_before_resolution: float = PROB_EXIT_BEFORE_RESOLUTION_DEFAULT,
    fill_probability: float | None = None,
    source: str | None = None,
    volume_24h: float | None = None,
    quote_age_sec: float | None = None,
) -> ScoreBreakdown:
    """Compute a proposal's A.5 score, fill probability, and adjusted size."""
    if side == "YES":
        ev_per_share = expected_value_per_share_yes(p_eff, ask)
    else:
        ev_per_share = expected_value_per_share_no(p_eff, bid)
    exit_cost = expected_exit_spread_cost_usd(
        spread, shares, prob_exit_before_resolution,
    )
    profit = expected_profit_usd(
        shares=shares,
        expected_value_per_share=ev_per_share,
        expected_exit_spread_cost=exit_cost,
        expected_slippage_cost=expected_slippage_cost,
        allocated_api_cost_usd=allocated_api_cost_usd,
    )
    cd = capital_days(size_usd, time_to_resolve_days)
    scarcity = total_scarcity_cost_usd(
        slots_used=slots_used,
        topic_exposure_usd=topic_exposure_usd,
        size_usd=size_usd,
        spent_today_usd=spent_today_usd,
        daily_hard_cap_usd=daily_hard_cap_usd,
    )
    score_raw = final_score(profit, cd, scarcity)
    if fill_probability is None:
        fp = fill_prob(
            source=source,
            volume_24h=volume_24h,
            spread=spread,
            shares=shares,
            quote_age_sec=quote_age_sec,
        )
    else:
        fp = max(0.0, min(1.0, fill_probability))
    # Final reported score keeps the deterministic A.5 number; callers
    # gate separately on `fp >= FILL_PROB_SKIP_THRESHOLD`. This matches
    # the spec, which treats fill_prob as a *separate* skip criterion.
    return ScoreBreakdown(
        expected_value_per_share=ev_per_share,
        expected_profit_usd=profit,
        capital_days_value=cd,
        scarcity_cost_usd=scarcity,
        score_raw=score_raw,
        fill_prob=fp,
        score=score_raw,
        fill_adjusted_size_usd=fill_adjusted_size_usd(size_usd, fp),
    )
