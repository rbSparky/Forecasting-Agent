"""Forecast configuration loader.

Reads the committed ``forecast_config.toml`` (or an override path via
``KALIBRE_FORECAST_CONFIG_PATH``) and exposes a frozen :class:`ForecastConfig`
that provider/runner code consults for:

- the default model identifier,
- per-million-token USD pricing,
- estimated input/output tokens for cost projection,
- the A.3 tier sigma values.

API keys are NEVER in this file; those stay in environment variables.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "forecast_config.toml"


@dataclass(frozen=True)
class ForecastConfig:
    default_model: str
    pricing: dict[str, tuple[float, float]]
    fallback_pricing: tuple[float, float]
    est_input_tokens: int
    est_output_tokens: int
    tier_sigma: dict[str, float]
    default_sigma: float
    source_path: str = ""

    def price_for(self, model_name: str) -> tuple[float, float]:
        """Return ``(input_per_mtok, output_per_mtok)`` for ``model_name``."""
        return self.pricing.get(model_name, self.fallback_pricing)

    def sigma_for_tier(self, tier: str | None) -> float:
        if not tier:
            return self.default_sigma
        return self.tier_sigma.get(tier.lower(), self.default_sigma)


def _coerce_pricing(table: dict[str, Any]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for model_id, entry in (table or {}).items():
        if not isinstance(entry, dict):
            continue
        try:
            inp = float(entry["input_per_mtok"])
            out = float(entry["output_per_mtok"])
        except (KeyError, TypeError, ValueError):
            continue
        result[model_id] = (inp, out)
    return result


def _coerce_tier_sigma(table: dict[str, Any]) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    default_value = 0.15
    for tier, value in (table or {}).items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        out[tier.lower()] = v
        if tier.lower() == "default":
            default_value = v
    return out, default_value


def load_forecast_config(
    path: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> ForecastConfig:
    """Load a :class:`ForecastConfig` from TOML.

    ``path`` overrides everything. If not supplied, we honor
    ``KALIBRE_FORECAST_CONFIG_PATH``; otherwise fall back to the
    committed default beside this module.
    """
    env_map = env if env is not None else os.environ
    chosen: Path
    if path is not None:
        chosen = Path(path)
    else:
        override = env_map.get("KALIBRE_FORECAST_CONFIG_PATH")
        chosen = Path(override) if override else DEFAULT_CONFIG_PATH
    if not chosen.exists():
        raise FileNotFoundError(f"forecast config not found at {chosen}")
    with chosen.open("rb") as f:
        data = tomllib.load(f)
    default_model = str(data.get("default_model", "openrouter:anthropic/claude-sonnet-4.6"))
    tokens = data.get("tokens") or {}
    est_input = int(tokens.get("est_input", 800))
    est_output = int(tokens.get("est_output", 300))
    fallback = data.get("fallback") or {}
    fallback_pricing = (
        float(fallback.get("input_per_mtok", 3.0)),
        float(fallback.get("output_per_mtok", 15.0)),
    )
    pricing = _coerce_pricing(data.get("pricing") or {})
    tier_sigma, default_sigma = _coerce_tier_sigma(data.get("tier_sigma") or {})
    return ForecastConfig(
        default_model=default_model,
        pricing=pricing,
        fallback_pricing=fallback_pricing,
        est_input_tokens=est_input,
        est_output_tokens=est_output,
        tier_sigma=tier_sigma,
        default_sigma=default_sigma,
        source_path=str(chosen),
    )


_DEFAULT_CONFIG: ForecastConfig | None = None


def get_default_config() -> ForecastConfig:
    """Memoized default config. Reset via :func:`reset_default_config` in tests."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load_forecast_config()
    return _DEFAULT_CONFIG


def reset_default_config() -> None:
    """Forget the cached default config (test helper)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None
