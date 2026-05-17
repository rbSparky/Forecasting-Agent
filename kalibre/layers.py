"""Phase 4A cheap-alpha layer configuration.

Reads the ``[layers.*]`` tables from ``kalibre/forecast/forecast_config.toml``
(or any path passed in / configured via ``KALIBRE_FORECAST_CONFIG_PATH``)
and exposes a frozen :class:`LayersConfig`. Keys live in environment
variables only; nothing in this file is secret.

Defaults are explicit and shadow-only:

- ``quote_history.enabled = true`` (features computed + persisted) but
  ``confidence_modifier_weight = deference_signal_weight =
  exit_signal_weight = 0.0`` so Phase 3B behavior is preserved.
- ``longshot.enabled = true`` (eligible candidates emit shadow rows) but
  ``primary_enabled = false`` so longshot priors never replace the
  blended LLM forecast unless explicitly flipped.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "forecast" / "forecast_config.toml"
)


@dataclass(frozen=True)
class QuoteHistoryLayer:
    enabled: bool = True
    confidence_modifier_weight: float = 0.0
    deference_signal_weight: float = 0.0
    exit_signal_weight: float = 0.0


@dataclass(frozen=True)
class LongshotLayer:
    enabled: bool = True
    primary_enabled: bool = False
    max_time_to_resolve_days: float = 7.0
    max_open_longshots: int = 5
    stop_loss_window: int = 20
    stop_loss_win_rate_threshold: float = 0.40
    stop_loss_suspension_hours: float = 24.0
    alpha: dict[str, float] = field(default_factory=dict)
    default_alpha: float = 1.0

    def alpha_for(self, category: str | None) -> float:
        if not category:
            return self.alpha.get("default", self.default_alpha)
        key = category.lower().strip()
        if key in self.alpha:
            return float(self.alpha[key])
        # Map common aliases.
        if key in ("price", "prices", "economy"):
            return float(self.alpha.get("prices", self.default_alpha))
        return float(self.alpha.get("default", self.default_alpha))


@dataclass(frozen=True)
class LayersConfig:
    quote_history: QuoteHistoryLayer = field(default_factory=QuoteHistoryLayer)
    longshot: LongshotLayer = field(default_factory=LongshotLayer)
    source_path: str = ""

    @property
    def any_qh_weight_nonzero(self) -> bool:
        qh = self.quote_history
        return (
            qh.confidence_modifier_weight != 0.0
            or qh.deference_signal_weight != 0.0
            or qh.exit_signal_weight != 0.0
        )


def _coerce_alpha_table(table: dict[str, Any]) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    default_alpha = 1.0
    for cat, value in (table or {}).items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        key = str(cat).lower().strip()
        out[key] = v
        if key == "default":
            default_alpha = v
    out.setdefault("default", default_alpha)
    return out, default_alpha


def _resolve_path(path: str | Path | None, env: dict[str, str] | None) -> Path:
    env_map = env if env is not None else os.environ
    if path is not None:
        return Path(path)
    override = env_map.get("KALIBRE_FORECAST_CONFIG_PATH")
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_layers_config(
    path: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> LayersConfig:
    """Load Phase 4A layer config. Returns dataclass defaults if file is missing."""
    chosen = _resolve_path(path, env)
    if not chosen.exists():
        return LayersConfig(source_path="")
    with chosen.open("rb") as f:
        data = tomllib.load(f)
    layers = data.get("layers") or {}
    qh_table = layers.get("quote_history") or {}
    long_table = layers.get("longshot") or {}
    quote_history = QuoteHistoryLayer(
        enabled=bool(qh_table.get("enabled", True)),
        confidence_modifier_weight=float(qh_table.get("confidence_modifier_weight", 0.0)),
        deference_signal_weight=float(qh_table.get("deference_signal_weight", 0.0)),
        exit_signal_weight=float(qh_table.get("exit_signal_weight", 0.0)),
    )
    alpha_table = long_table.get("alpha") or {}
    alpha_map, default_alpha = _coerce_alpha_table(alpha_table)
    longshot = LongshotLayer(
        enabled=bool(long_table.get("enabled", True)),
        primary_enabled=bool(long_table.get("primary_enabled", False)),
        max_time_to_resolve_days=float(long_table.get("max_time_to_resolve_days", 7.0)),
        max_open_longshots=int(long_table.get("max_open_longshots", 5)),
        stop_loss_window=int(long_table.get("stop_loss_window", 20)),
        stop_loss_win_rate_threshold=float(long_table.get("stop_loss_win_rate_threshold", 0.40)),
        stop_loss_suspension_hours=float(long_table.get("stop_loss_suspension_hours", 24.0)),
        alpha=alpha_map,
        default_alpha=default_alpha,
    )
    return LayersConfig(
        quote_history=quote_history, longshot=longshot, source_path=str(chosen),
    )


_DEFAULT_LAYERS: LayersConfig | None = None


def get_default_layers_config() -> LayersConfig:
    global _DEFAULT_LAYERS
    if _DEFAULT_LAYERS is None:
        _DEFAULT_LAYERS = load_layers_config()
    return _DEFAULT_LAYERS


def reset_default_layers_config() -> None:
    """Test helper: forget the cached default."""
    global _DEFAULT_LAYERS
    _DEFAULT_LAYERS = None
