"""Phase 6D budget profiles.

A budget profile bundles the spend-shaping knobs that make Kalibre
viable under a specific dollar/timeline target. Two profiles ship:

- ``standard`` (default) — preserves Phase 6B/6C behaviour. Total
  budget $500, daily hard cap $35, no per-tick paid forecast cap,
  web-search max_results=4 / max_total_results=12, exploration
  selector max=3, web-search cache TTL 30 min, Opus default on.

- ``micro`` — designed for a $50 / 14-day evaluation window. Total
  budget $50, daily hard cap $3.25, daily soft cap $2.50, per-tick
  paid forecast cap $0.05, web-search daily hard cap $1.00,
  web-search max_results=2 / max_total_results=4, exploration
  selector max=1, web-search cache TTL 2h, Opus default OFF.

The profile is a frozen dataclass with one ``from_env`` classmethod;
operators flip profiles via ``KALIBRE_BUDGET_PROFILE=standard|micro``.
Individual knobs can still be overridden by their existing env vars
(``KALIBRE_WEB_SEARCH_MAX_RESULTS`` etc.) -- the profile provides
*defaults* the per-knob env reader falls back on when the env var
is unset, not a hard override.

The runner reads the profile once per pass to gate Opus and to apply
the per-tick paid cap. The loop reads it once at startup to size the
SpendGovernor + cache + selector + provider configs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


PROFILE_STANDARD = "standard"
PROFILE_MICRO = "micro"

# Standard profile mirrors the pre-Phase-6D behaviour.
STD_TOTAL_BUDGET_USD = 500.0
STD_DAILY_HARD_USD = 35.0
STD_DAILY_SOFT_USD = 25.0
STD_PER_TICK_PAID_CAP_USD = 1e9  # effectively off
STD_WEB_SEARCH_DAILY_HARD_USD = 10.0
STD_WEB_SEARCH_MAX_RESULTS = 4
STD_WEB_SEARCH_MAX_TOTAL_RESULTS = 12
STD_SELECTOR_EXPLORATION_MAX = 3
STD_CACHE_TTL_SEC = 30 * 60
STD_CACHE_TTL_SHORT_RESOLUTION_SEC = 30 * 60
STD_OPUS_DEFAULT_ON = True

# Micro profile sized for $50 / 14-day evaluation.
MICRO_TOTAL_BUDGET_USD = 50.0
MICRO_DAILY_HARD_USD = 3.25
MICRO_DAILY_SOFT_USD = 2.50
MICRO_PER_TICK_PAID_CAP_USD = 0.05
MICRO_WEB_SEARCH_DAILY_HARD_USD = 1.00
MICRO_WEB_SEARCH_MAX_RESULTS = 2
MICRO_WEB_SEARCH_MAX_TOTAL_RESULTS = 4
MICRO_SELECTOR_EXPLORATION_MAX = 1
MICRO_CACHE_TTL_SEC = 2 * 3600
MICRO_CACHE_TTL_SHORT_RESOLUTION_SEC = 30 * 60
MICRO_SHORT_RESOLUTION_HOURS = 6.0
MICRO_OPUS_DEFAULT_ON = False
# 14-day evaluation window used by projected-spend reports.
DEFAULT_EVAL_WINDOW_DAYS = 14


@dataclass(frozen=True)
class BudgetProfile:
    """Spend-shaping knobs that vary together with the budget target.

    The runner reads ``opus_default_on``, ``per_tick_paid_forecast_cap_usd``,
    ``cache_ttl_sec``, ``cache_ttl_short_resolution_sec``, and
    ``short_resolution_threshold_hours`` directly; the loop reads the
    SpendGovernor / web-search / selector defaults at startup. The
    profile's defaults are overridable by their existing env vars --
    callers should treat profile fields as "default if env var unset",
    NOT as authoritative overrides.
    """

    name: str
    total_budget_usd: float
    daily_hard_usd: float
    daily_soft_usd: float
    per_tick_paid_forecast_cap_usd: float
    web_search_daily_hard_cap_usd: float
    web_search_max_results: int
    web_search_max_total_results: int
    selector_exploration_max: int
    cache_ttl_sec: int
    cache_ttl_short_resolution_sec: int
    short_resolution_threshold_hours: float
    opus_default_on: bool
    eval_window_days: int = DEFAULT_EVAL_WINDOW_DAYS

    @property
    def is_micro(self) -> bool:
        return self.name == PROFILE_MICRO

    @classmethod
    def standard(cls) -> "BudgetProfile":
        return cls(
            name=PROFILE_STANDARD,
            total_budget_usd=STD_TOTAL_BUDGET_USD,
            daily_hard_usd=STD_DAILY_HARD_USD,
            daily_soft_usd=STD_DAILY_SOFT_USD,
            per_tick_paid_forecast_cap_usd=STD_PER_TICK_PAID_CAP_USD,
            web_search_daily_hard_cap_usd=STD_WEB_SEARCH_DAILY_HARD_USD,
            web_search_max_results=STD_WEB_SEARCH_MAX_RESULTS,
            web_search_max_total_results=STD_WEB_SEARCH_MAX_TOTAL_RESULTS,
            selector_exploration_max=STD_SELECTOR_EXPLORATION_MAX,
            cache_ttl_sec=STD_CACHE_TTL_SEC,
            cache_ttl_short_resolution_sec=STD_CACHE_TTL_SHORT_RESOLUTION_SEC,
            short_resolution_threshold_hours=MICRO_SHORT_RESOLUTION_HOURS,
            opus_default_on=STD_OPUS_DEFAULT_ON,
        )

    @classmethod
    def micro(cls) -> "BudgetProfile":
        return cls(
            name=PROFILE_MICRO,
            total_budget_usd=MICRO_TOTAL_BUDGET_USD,
            daily_hard_usd=MICRO_DAILY_HARD_USD,
            daily_soft_usd=MICRO_DAILY_SOFT_USD,
            per_tick_paid_forecast_cap_usd=MICRO_PER_TICK_PAID_CAP_USD,
            web_search_daily_hard_cap_usd=MICRO_WEB_SEARCH_DAILY_HARD_USD,
            web_search_max_results=MICRO_WEB_SEARCH_MAX_RESULTS,
            web_search_max_total_results=MICRO_WEB_SEARCH_MAX_TOTAL_RESULTS,
            selector_exploration_max=MICRO_SELECTOR_EXPLORATION_MAX,
            cache_ttl_sec=MICRO_CACHE_TTL_SEC,
            cache_ttl_short_resolution_sec=MICRO_CACHE_TTL_SHORT_RESOLUTION_SEC,
            short_resolution_threshold_hours=MICRO_SHORT_RESOLUTION_HOURS,
            opus_default_on=MICRO_OPUS_DEFAULT_ON,
        )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BudgetProfile":
        env_map = env if env is not None else dict(os.environ)
        name = (env_map.get("KALIBRE_BUDGET_PROFILE") or PROFILE_STANDARD).strip().lower()
        if name == PROFILE_MICRO:
            return cls.micro()
        return cls.standard()


def force_opus_enabled(env: dict[str, str] | None = None) -> bool:
    """Return True if the operator has explicitly opted into Opus calls
    via ``KALIBRE_FORCE_OPUS=1``. Used under the micro profile to
    let an operator override the default-off Opus policy for a single
    run (e.g. an interactive validation tick)."""
    env_map = env if env is not None else dict(os.environ)
    return (env_map.get("KALIBRE_FORCE_OPUS") or "").strip() == "1"


def project_14_day_spend_usd(
    *,
    spend_total_usd: float,
    ticks_observed: int,
    eval_window_days: int = DEFAULT_EVAL_WINDOW_DAYS,
    ticks_per_day: int = 96,
) -> float:
    """Linear extrapolation of total spend to a 14-day window.

    ``ticks_per_day`` defaults to 96 (the loop's 15-minute cadence).
    A zero ``ticks_observed`` collapses to ``spent_total`` (no
    extrapolation possible). The estimate is rough -- ticks where the
    universe is empty contribute zero, ticks with heavy evidence pulls
    contribute outliers -- but it's enough to flag "we're on track to
    blow $50" before it happens.
    """
    if ticks_observed <= 0:
        return float(spend_total_usd)
    per_tick_avg = float(spend_total_usd) / float(ticks_observed)
    return per_tick_avg * float(ticks_per_day) * float(eval_window_days)
