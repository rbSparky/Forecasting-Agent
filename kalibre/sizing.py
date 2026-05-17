"""Pessimistic fractional Kelly sizing (AGENTS.md A.1-A.4).

All pure functions. ``size_proposal`` orchestrates the whole pipeline:

1. A.1 Kelly fractions for BUY YES and BUY NO.
2. A.2 pessimism: ``p_eff`` shifts away from the side we'd take.
3. A.3 sigma-p clipping into ``[0.02, 0.25]``.
4. A.4 fractional-Kelly ramp, multipliers, final clip, min-trade gate.

The caller supplies the raw forecast ``p_mean`` and ``sigma_p``. We do
not produce probabilities here -- that's the LLM/calibration layer's job
and Phase 2 deliberately does not implement it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# --- thresholds -------------------------------------------------------------

MIN_TRADE_USD = 25.0
MAX_INITIAL_SINGLE_MARKET_USD = 300.0
SERVER_PER_MARKET_CAP_USD = 1000.0

SIGMA_P_LO = 0.02
SIGMA_P_HI = 0.25

DEFAULT_Z_COLD = 0.5
DEFAULT_Z_WARM = 0.25
COLD_START_N = 30
WARM_N = 100


# --- A.1: Kelly fractions ---------------------------------------------------


def kelly_yes(p: float, ask: float) -> float:
    """BUY YES Kelly fraction: ``max(0, (p - ask) / (1 - ask))``."""
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    return max(0.0, (p - ask) / (1.0 - ask))


def kelly_no(p: float, bid: float) -> float:
    """BUY NO Kelly fraction: ``max(0, (bid - p) / bid)``."""
    if bid <= 0.0 or bid >= 1.0:
        return 0.0
    return max(0.0, (bid - p) / bid)


# --- A.2: pessimism ---------------------------------------------------------


def p_eff_for_side(p_mean: float, sigma_p: float, side: Literal["YES", "NO"], z: float) -> float:
    """Pessimism-adjusted probability for the side being considered.

    For BUY YES we shift ``p_mean`` *down*; for BUY NO we shift *up*. Both
    are clipped to ``[0.02, 0.98]`` so Kelly stays well-defined.
    """
    if sigma_p < 0.0:
        sigma_p = 0.0
    if side == "YES":
        return max(0.02, p_mean - z * sigma_p)
    return min(0.98, p_mean + z * sigma_p)


def pessimism_z(n_resolved: int) -> float:
    """``z`` ramps from 0.5 (cold) down to 0.25 (warm)."""
    if n_resolved >= WARM_N:
        return DEFAULT_Z_WARM
    if n_resolved >= COLD_START_N:
        return 0.4
    return DEFAULT_Z_COLD


# --- A.3: sigma-p model -----------------------------------------------------


_TIER_SIGMA = {
    "opus": 0.05,
    "sonnet": 0.08,
    "haiku": 0.12,
    "structural_only": 0.15,
    "external": 0.10,
}


def sigma_horizon(time_to_resolve_hours: float) -> float:
    if time_to_resolve_hours < 6:
        return 0.0
    if time_to_resolve_hours < 24:
        return 0.02
    if time_to_resolve_hours < 24 * 7:
        return 0.04
    return 0.07


def sigma_cal(n_bucket: int) -> float:
    return 0.10 / math.sqrt(max(0, n_bucket) + 5)


def sigma_tier(model_tier: str) -> float:
    return _TIER_SIGMA.get((model_tier or "").lower(), _TIER_SIGMA["external"])


def clip_sigma_p(value: float) -> float:
    """Clip ``value`` into the documented ``[0.02, 0.25]`` range."""
    if value != value:  # NaN
        return SIGMA_P_LO
    return min(SIGMA_P_HI, max(SIGMA_P_LO, value))


def compute_sigma_p(
    *,
    model_tier: str = "external",
    n_bucket: int = 0,
    time_to_resolve_hours: float = 24.0,
    sigma_ensemble: float = 0.0,
    dual_channel_delta: float = 0.0,
) -> float:
    """Combine the variance components from A.3 and clip into the safe range."""
    sigma_source = 0.30 * max(0.0, dual_channel_delta)
    var = (
        max(0.0, sigma_ensemble) ** 2
        + sigma_cal(n_bucket) ** 2
        + sigma_tier(model_tier) ** 2
        + sigma_horizon(time_to_resolve_hours) ** 2
        + sigma_source ** 2
    )
    return clip_sigma_p(math.sqrt(var))


# --- A.4: multipliers, fractional Kelly, final size -------------------------


def fractional_kelly_multiplier(n_resolved: int) -> float:
    if n_resolved >= WARM_N:
        return 0.20
    if n_resolved >= COLD_START_N:
        return 0.15
    return 0.10


def confidence_mult(sigma_p: float) -> float:
    return min(1.0, max(0.1, 1.0 - 4.0 * sigma_p * sigma_p))


def diversification_mult(topic_exposure_usd: float) -> float:
    return min(1.0, max(0.3, 1.0 - max(0.0, topic_exposure_usd) / 2000.0))


def drawdown_mult(pnl_24h: float) -> float:
    return 0.5 if pnl_24h < -1000.0 else 1.0


# --- proposal output --------------------------------------------------------


@dataclass(frozen=True)
class SizingInputs:
    """Everything :func:`size_proposal` needs."""

    market_id: str
    best_bid: float
    best_ask: float
    p_mean: float
    sigma_p: float
    bankroll_usd: float
    exposure_market_usd: float = 0.0
    topic_exposure_usd: float = 0.0
    pnl_24h: float = 0.0
    remaining_gross_cap_usd: float = 10_000.0
    n_resolved: int = 0
    z: float | None = None
    max_initial_single_market_usd: float = MAX_INITIAL_SINGLE_MARKET_USD
    server_per_market_cap_usd: float = SERVER_PER_MARKET_CAP_USD


@dataclass(frozen=True)
class SizingResult:
    market_id: str
    action: str  # 'BUY' | 'HOLD'
    side: str | None  # 'YES' | 'NO' | None
    size_usd: float
    shares: float
    raw_kelly_f: float
    fractional_kelly: float
    confidence_mult: float
    diversification_mult: float
    drawdown_mult: float
    p_mean: float
    p_eff: float
    sigma_p: float
    z: float
    decision: str  # 'propose' | 'hold'
    reject_reason: str | None
    debug: dict[str, float] = field(default_factory=dict)


def _hold(
    inputs: SizingInputs,
    *,
    reason: str,
    p_eff: float = 0.0,
    sigma_p: float | None = None,
    z: float | None = None,
    raw_kelly_f: float = 0.0,
    confidence: float = 1.0,
    diversification: float = 1.0,
    drawdown: float = 1.0,
    side: str | None = None,
) -> SizingResult:
    return SizingResult(
        market_id=inputs.market_id,
        action="HOLD",
        side=side,
        size_usd=0.0,
        shares=0.0,
        raw_kelly_f=raw_kelly_f,
        fractional_kelly=fractional_kelly_multiplier(inputs.n_resolved),
        confidence_mult=confidence,
        diversification_mult=diversification,
        drawdown_mult=drawdown,
        p_mean=inputs.p_mean,
        p_eff=p_eff,
        sigma_p=(sigma_p if sigma_p is not None else clip_sigma_p(inputs.sigma_p)),
        z=z if z is not None else (inputs.z if inputs.z is not None else pessimism_z(inputs.n_resolved)),
        decision="hold",
        reject_reason=reason,
        debug={},
    )


def size_proposal(inputs: SizingInputs) -> SizingResult:
    """Run the full A.1-A.4 pipeline and return a sizing decision."""
    sigma_p = clip_sigma_p(inputs.sigma_p)
    z = inputs.z if inputs.z is not None else pessimism_z(inputs.n_resolved)

    if not (0.0 <= inputs.p_mean <= 1.0):
        return _hold(inputs, reason="invalid_p_mean", sigma_p=sigma_p, z=z)
    bid, ask = inputs.best_bid, inputs.best_ask
    if not (0.0 < bid < 1.0) or not (0.0 < ask < 1.0) or ask < bid:
        return _hold(inputs, reason="invalid_quote", sigma_p=sigma_p, z=z)

    p_eff_yes = p_eff_for_side(inputs.p_mean, sigma_p, "YES", z)
    p_eff_no = p_eff_for_side(inputs.p_mean, sigma_p, "NO", z)
    f_yes = kelly_yes(p_eff_yes, ask)
    f_no = kelly_no(p_eff_no, bid)

    if f_yes <= 0.0 and f_no <= 0.0:
        return _hold(
            inputs,
            reason="kelly_zero",
            sigma_p=sigma_p,
            z=z,
            p_eff=(p_eff_yes if inputs.p_mean >= ask else p_eff_no),
        )

    side: Literal["YES", "NO"]
    if f_yes >= f_no:
        side, raw_f, p_eff = "YES", f_yes, p_eff_yes
        side_price = ask
    else:
        side, raw_f, p_eff = "NO", f_no, p_eff_no
        side_price = 1.0 - bid

    conf = confidence_mult(sigma_p)
    div = diversification_mult(inputs.topic_exposure_usd)
    dd = drawdown_mult(inputs.pnl_24h)
    frac = fractional_kelly_multiplier(inputs.n_resolved)

    raw_size_usd = raw_f * frac * max(0.0, inputs.bankroll_usd)
    pre_clip_size_usd = raw_size_usd * conf * div * dd
    # A.4: hi = min($300, $1000 - exposure_market, remaining_gross_cap).
    per_market_remaining = max(
        0.0,
        inputs.server_per_market_cap_usd - max(0.0, inputs.exposure_market_usd),
    )
    upper = min(
        inputs.max_initial_single_market_usd,
        per_market_remaining,
        max(0.0, inputs.remaining_gross_cap_usd),
    )
    size_usd = min(pre_clip_size_usd, upper)
    if size_usd < MIN_TRADE_USD:
        return _hold(
            inputs,
            reason=("size_below_min" if pre_clip_size_usd > 0 else "kelly_zero"),
            sigma_p=sigma_p,
            z=z,
            p_eff=p_eff,
            raw_kelly_f=raw_f,
            confidence=conf,
            diversification=div,
            drawdown=dd,
            side=side,
        )

    shares = size_usd / max(1e-9, side_price)
    return SizingResult(
        market_id=inputs.market_id,
        action="BUY",
        side=side,
        size_usd=size_usd,
        shares=shares,
        raw_kelly_f=raw_f,
        fractional_kelly=frac,
        confidence_mult=conf,
        diversification_mult=div,
        drawdown_mult=dd,
        p_mean=inputs.p_mean,
        p_eff=p_eff,
        sigma_p=sigma_p,
        z=z,
        decision="propose",
        reject_reason=None,
        debug={
            "side_price": side_price,
            "raw_size_usd": raw_size_usd,
            "pre_clip_size_usd": pre_clip_size_usd,
            "per_market_remaining": per_market_remaining,
            "upper_clip": upper,
        },
    )
