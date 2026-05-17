"""Phase 3 forecast layer.

Pure-function prompt builder + JSON parser, an OpenAI-compatible HTTP
provider, an SQLite-backed cache, and a per-tick runner that ties them
together. Outputs feed Phase 2's deterministic risk engine via
:class:`kalibre.strategy.MarketProbability`.

The runner respects the spend governor, the local T+480s submit cutoff,
and the per-tick forecast cap.
"""

from kalibre.forecast.cache import (
    ForecastCache,
    MemoryForecastCache,
    SQLiteForecastCache,
    cache_key,
)
from kalibre.forecast.halawi import (
    JSON_INSTRUCTION,
    build_prompt,
    parse_forecast_json,
)
from kalibre.forecast.provider import (
    ForecastProvider,
    HTTPForecastProvider,
    NoOpForecastProvider,
    ProviderConfig,
    build_default_provider,
)
from kalibre.forecast.runner import (
    ForecastPassResult,
    forecast_audit_row,
    run_forecast_pass,
)
from kalibre.forecast.types import (
    ForecastBatchResult,
    ForecastRequest,
    ForecastResult,
)

__all__ = [
    "ForecastBatchResult",
    "ForecastCache",
    "ForecastPassResult",
    "ForecastProvider",
    "ForecastRequest",
    "ForecastResult",
    "HTTPForecastProvider",
    "JSON_INSTRUCTION",
    "MemoryForecastCache",
    "NoOpForecastProvider",
    "ProviderConfig",
    "SQLiteForecastCache",
    "build_default_provider",
    "build_prompt",
    "cache_key",
    "forecast_audit_row",
    "parse_forecast_json",
    "run_forecast_pass",
]
