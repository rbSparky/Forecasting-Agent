"""Market-implied prior blending (Phase 3B).

Pure-function shrinkage that combines the calibrated LLM forecast with
the market-implied prior. The blend lives between forecasting and
sizing: the runner produces a calibrated model probability, the strategy
consumes ``p_blend`` so the deterministic risk engine never sees a raw
model forecast.

Design (deterministic, documented, bounded):

- ``p_market`` is the quote mid clipped to ``[0.02, 0.98]``.
- ``sigma_model`` comes from the existing A.3 ladder.
- ``sigma_market`` is chosen from four regimes:

    1. ``near_resolution`` (time-to-resolve < 2h) -> 0.03; market dominates.
    2. ``tight_smart_market`` (spread <= 0.02 AND volume >= $5K) -> 0.05.
    3. ``wide_noisy_market`` (spread > 0.05 OR volume < $500) -> 0.20.
    4. ``default``                                            -> 0.10.

  This is intentionally discrete: every audit row records the chosen
  ``blend_reason`` so post-hoc analysis can split realized PnL by regime.

- Weights are standard inverse-variance:

    w_model  = (1/sigma_model^2) / (1/sigma_model^2 + 1/sigma_market^2)
    w_market = 1 - w_model
    sigma_blend = sqrt(1 / (1/sigma_model^2 + 1/sigma_market^2))

- ``optional_extra_shrinkage`` (default 0.0) leaves an explicit lever for
  future T2 alpha (quote-history smart-market boost) WITHOUT changing
  the blend signature later. It biases the weight pair toward the market
  by ``extra * (1 - market_weight)``.

- All probabilities (model, market, blend) and both sigmas are clipped
  into the documented safe ranges before returning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


P_MIN = 0.02
P_MAX = 0.98
SIGMA_MIN = 0.02
SIGMA_MAX = 0.25

REASON_NEAR_RESOLUTION = "near_resolution"
REASON_TIGHT_SMART = "tight_smart_market"
REASON_WIDE_NOISY = "wide_noisy_market"
REASON_DEFAULT = "default"

# Hyper-parameters of the four-regime market-sigma policy. Knobs are
# exposed as module constants so the next phase can promote them into
# the forecast TOML config without touching call sites.
SIGMA_MARKET_NEAR_RESOLUTION = 0.03
SIGMA_MARKET_TIGHT_SMART = 0.05
SIGMA_MARKET_WIDE_NOISY = 0.20
SIGMA_MARKET_DEFAULT = 0.10

TIGHT_SPREAD = 0.02
SMART_VOLUME_USD = 5_000.0
WIDE_SPREAD = 0.05
THIN_VOLUME_USD = 500.0
NEAR_RESOLUTION_HOURS = 2.0


@dataclass(frozen=True)
class BlendResult:
    """Per-market blend outcome. Used for both the MarketProbability and audit."""

    p_model_cal: float
    p_market: float
    p_blend: float
    market_weight: float
    model_weight: float
    sigma_model: float
    sigma_market: float
    sigma_blend: float
    blend_reason: str

    def to_audit_dict(self) -> dict[str, float | str]:
        """Serialize the fields the audit_json must carry."""
        return {
            "p_model_cal": self.p_model_cal,
            "p_market": self.p_market,
            "p_blend": self.p_blend,
            "market_weight": self.market_weight,
            "model_weight": self.model_weight,
            "sigma_model": self.sigma_model,
            "sigma_market": self.sigma_market,
            "sigma_blend": self.sigma_blend,
            "blend_reason": self.blend_reason,
        }


# --- helpers ---------------------------------------------------------------


def _clip(value: float, lo: float, hi: float) -> float:
    if value != value:  # NaN
        return (lo + hi) / 2.0
    return min(hi, max(lo, value))


def compute_market_prior(
    *,
    best_bid: float | None,
    best_ask: float | None,
) -> float | None:
    """Quote mid clipped to ``[0.02, 0.98]``. Returns ``None`` if unusable."""
    if best_bid is None or best_ask is None:
        return None
    if best_ask < best_bid:
        return None
    if best_bid <= 0 or best_ask <= 0:
        return None
    mid = 0.5 * (float(best_bid) + float(best_ask))
    if mid <= 0 or mid >= 1:
        # Quote is structurally broken; refuse a prior.
        return None
    return _clip(mid, P_MIN, P_MAX)


def market_sigma(
    *,
    spread: float | None,
    volume_24h: float | None,
    time_to_resolve_hours: float | None,
) -> tuple[float, str]:
    """Select the regime sigma and the human-readable reason."""
    if time_to_resolve_hours is not None and time_to_resolve_hours < NEAR_RESOLUTION_HOURS:
        return SIGMA_MARKET_NEAR_RESOLUTION, REASON_NEAR_RESOLUTION
    tight = spread is not None and spread <= TIGHT_SPREAD
    smart_volume = volume_24h is not None and volume_24h >= SMART_VOLUME_USD
    if tight and smart_volume:
        return SIGMA_MARKET_TIGHT_SMART, REASON_TIGHT_SMART
    wide = spread is not None and spread > WIDE_SPREAD
    thin = volume_24h is not None and volume_24h < THIN_VOLUME_USD
    if wide or thin:
        return SIGMA_MARKET_WIDE_NOISY, REASON_WIDE_NOISY
    return SIGMA_MARKET_DEFAULT, REASON_DEFAULT


def _inverse_variance(sigma_a: float, sigma_b: float) -> tuple[float, float, float]:
    """Inverse-variance weighted average components.

    Returns ``(w_a, w_b, sigma_blend)`` where ``sigma_blend`` is
    ``sqrt(1 / (1/sigma_a^2 + 1/sigma_b^2))``.
    """
    var_a = sigma_a * sigma_a
    var_b = sigma_b * sigma_b
    inv = 1.0 / var_a + 1.0 / var_b
    w_a = (1.0 / var_a) / inv
    w_b = (1.0 / var_b) / inv
    sigma_blend = math.sqrt(1.0 / inv)
    return w_a, w_b, sigma_blend


def inverse_variance_blend(
    p_a: float, sigma_a: float, p_b: float, sigma_b: float,
) -> tuple[float, float]:
    """Public Phase 6 helper: inverse-variance blend of two estimates.

    Returns ``(p_blend, sigma_blend)`` both clipped into ``[P_MIN, P_MAX]``
    / ``[SIGMA_MIN, SIGMA_MAX]``. Used by the Opus escalation route to
    combine Sonnet and Opus probabilities. Uses the same math as the
    Phase 3B market-prior blend so the audit story stays uniform.
    """
    sa = _clip(float(sigma_a), SIGMA_MIN, SIGMA_MAX)
    sb = _clip(float(sigma_b), SIGMA_MIN, SIGMA_MAX)
    pa = _clip(float(p_a), P_MIN, P_MAX)
    pb = _clip(float(p_b), P_MIN, P_MAX)
    w_a, w_b, sigma_blend = _inverse_variance(sa, sb)
    p_blend = _clip(w_a * pa + w_b * pb, P_MIN, P_MAX)
    sigma_blend = _clip(sigma_blend, SIGMA_MIN, SIGMA_MAX)
    return p_blend, sigma_blend


# --- main blender ---------------------------------------------------------


def blend_with_market_prior(
    *,
    p_model_cal: float,
    sigma_model: float,
    p_market: float | None,
    spread: float | None,
    volume_24h: float | None,
    time_to_resolve_hours: float | None,
    model_tier: str | None = None,
    optional_extra_shrinkage: float = 0.0,
) -> BlendResult:
    """Blend the calibrated model probability with the market-implied prior.

    If ``p_market`` is unavailable, the model carries the full weight and
    ``blend_reason`` is reported as ``"default"`` with
    ``market_weight=0``. Callers should normally not invoke this with
    ``p_market=None`` -- :func:`compute_market_prior` is the gate.
    """
    p_model = _clip(float(p_model_cal), P_MIN, P_MAX)
    sigma_m = _clip(float(sigma_model), SIGMA_MIN, SIGMA_MAX)

    if p_market is None:
        return BlendResult(
            p_model_cal=p_model,
            p_market=p_model,  # echo; no real prior available
            p_blend=p_model,
            market_weight=0.0,
            model_weight=1.0,
            sigma_model=sigma_m,
            sigma_market=SIGMA_MAX,
            sigma_blend=sigma_m,
            blend_reason=REASON_DEFAULT,
        )
    p_mkt = _clip(float(p_market), P_MIN, P_MAX)
    sigma_mkt_raw, reason = market_sigma(
        spread=spread, volume_24h=volume_24h, time_to_resolve_hours=time_to_resolve_hours,
    )
    sigma_mkt = _clip(sigma_mkt_raw, SIGMA_MIN, SIGMA_MAX)

    w_model, w_market, sigma_blend = _inverse_variance(sigma_m, sigma_mkt)

    extra = _clip(float(optional_extra_shrinkage), 0.0, 0.5)
    if extra > 0.0:
        # Pull both weights toward the market by `extra * (1 - market_weight)`.
        w_market = w_market + extra * (1.0 - w_market)
        w_model = 1.0 - w_market

    p_blend = _clip(w_model * p_model + w_market * p_mkt, P_MIN, P_MAX)
    sigma_blend = _clip(sigma_blend, SIGMA_MIN, SIGMA_MAX)
    return BlendResult(
        p_model_cal=p_model,
        p_market=p_mkt,
        p_blend=p_blend,
        market_weight=w_market,
        model_weight=w_model,
        sigma_model=sigma_m,
        sigma_market=sigma_mkt,
        sigma_blend=sigma_blend,
        blend_reason=reason,
    )
