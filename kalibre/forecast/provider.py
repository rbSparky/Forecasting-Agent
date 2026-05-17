"""OpenAI-compatible forecast provider.

This module is deliberately thin: it issues a single chat-completion
HTTP call per market, parses the JSON the model emits, and returns a
:class:`ForecastResult`. Errors never raise out -- they're packed into
``ForecastResult.error`` so the caller can record an audit row and move
on without crashing the tick.

Configuration:

- Pricing, default model id, estimated token usage, and tier sigmas
  live in ``kalibre/forecast/forecast_config.toml`` (see
  :mod:`kalibre.forecast.config`). The file is committed and contains
  no secrets.
- Override the default config path with
  ``KALIBRE_FORECAST_CONFIG_PATH``. Override the default model with
  ``KALIBRE_FORECAST_MODEL`` (any value present in ``[pricing.*]`` or a
  pass-through model id).
- API keys are *only* in environment variables:
  ``OPENROUTER_API_KEY`` (+ ``OPENROUTER_BASE_URL``) or
  ``OPENAI_API_KEY`` (+ ``OPENAI_BASE_URL``). Defaults are
  ``https://openrouter.ai/api/v1`` and ``https://api.openai.com/v1``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from kalibre.forecast.config import ForecastConfig, get_default_config
from kalibre.forecast.halawi import (
    ForecastParseError,
    build_prompt,
    parse_forecast_json,
)
from kalibre.forecast.types import ForecastRequest, ForecastResult

logger = logging.getLogger("kalibre.forecast.provider")


DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


def _classify_model(model: str) -> tuple[str, str]:
    """Return ``(api_provider, model_name)``.

    ``api_provider`` is the prefix we use to pick the API key + base URL.
    Recognized prefixes: ``openrouter:``, ``openai:``. Anything else is
    treated as ``"openai"`` (the most permissive shape).
    """
    if ":" in model:
        prefix, name = model.split(":", 1)
        prefix = prefix.lower()
        if prefix in ("openrouter", "openai"):
            return prefix, name
    return "openai", model


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    config: ForecastConfig | None = None,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Estimate the per-call USD cost.

    Backward-compatible: callers can pass ``pricing`` directly for an
    inline override, or ``config`` to switch the entire table (default
    model, fallback rates, token estimates). With nothing supplied we
    use the committed default config.
    """
    cfg = config or get_default_config()
    inp = cfg.est_input_tokens if input_tokens is None else int(input_tokens)
    out = cfg.est_output_tokens if output_tokens is None else int(output_tokens)
    _, name = _classify_model(model)
    if pricing is not None:
        p_in, p_out = pricing.get(name, cfg.fallback_pricing)
    else:
        p_in, p_out = cfg.price_for(name)
    return (inp / 1_000_000.0) * p_in + (out / 1_000_000.0) * p_out


def _model_tier_from_name(name: str) -> str:
    n = name.lower()
    if "opus" in n:
        return "opus"
    if "sonnet" in n:
        return "sonnet"
    if "haiku" in n:
        return "haiku"
    if "gpt-4o" in n:
        return "sonnet"  # rough cost-tier mapping
    return "external"


# --- provider config + Protocol --------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    api_key: str
    base_url: str
    edge_source: str = "sonnet_forecast"
    model_tier: str = "sonnet"
    temperature: float = 0.0
    max_tokens: int = 600
    http_timeout: float = 30.0


class ForecastProvider(Protocol):
    """Single-market forecast provider. Errors live in ``ForecastResult``."""

    @property
    def model(self) -> str: ...

    def forecast_one(self, request: ForecastRequest) -> ForecastResult: ...


# --- HTTP provider ---------------------------------------------------------


class HTTPForecastProvider:
    """OpenAI-compatible HTTP provider built on httpx.

    Never raises on call failure -- network errors, timeouts, non-200
    responses, and unparseable bodies are returned as
    :class:`ForecastResult` with ``error`` set.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: httpx.Client | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        forecast_config: ForecastConfig | None = None,
    ) -> None:
        self.config = config
        self.forecast_config = forecast_config or get_default_config()
        # Caller-supplied ``pricing`` still wins (kept for backwards-compat
        # in tests); otherwise the price book comes from forecast_config.
        self.pricing = pricing if pricing is not None else self.forecast_config.pricing
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=config.base_url.rstrip("/"),
                timeout=config.http_timeout,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                http2=False,
            )
            self._owns_client = True

    @property
    def model(self) -> str:
        return self.config.model

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def forecast_one(self, request: ForecastRequest) -> ForecastResult:
        result = ForecastResult(
            market_id=request.market_id,
            model=self.config.model,
            model_tier=self.config.model_tier,
            edge_source=self.config.edge_source,
        )
        _, model_name = _classify_model(self.config.model)
        body: dict[str, Any] = {
            "model": model_name,
            "messages": build_prompt(request),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        started = time.monotonic()
        try:
            response = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            result.error = f"timeout: {exc}"
            return result
        except httpx.HTTPError as exc:
            result.error = f"http_error: {type(exc).__name__}"
            return result
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if response.status_code == 401:
            result.error = "unauthorized"
            return result
        if response.status_code == 429:
            result.error = "rate_limited"
            return result
        if response.status_code >= 400:
            body_preview = (response.text or "")[:240]
            result.error = f"http_{response.status_code}: {body_preview}"
            return result
        try:
            payload = response.json()
        except ValueError as exc:
            result.error = f"invalid_json: {exc}"
            return result
        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            result.error = f"unexpected_response_shape: {exc}"
            return result
        try:
            parsed = parse_forecast_json(content)
        except ForecastParseError as exc:
            result.error = f"parse_error: {exc}"
            return result
        result.p_raw = float(parsed["p_yes"])
        result.p_mean = result.p_raw  # callers calibrate later
        result.confidence = parsed.get("confidence")
        result.rationale = parsed.get("rationale") or ""
        usage = payload.get("usage") or {}
        result.input_tokens = int(usage.get("prompt_tokens") or 0)
        result.output_tokens = int(usage.get("completion_tokens") or 0)
        result.cached_tokens = int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )
        result.api_cost_usd = estimate_cost_usd(
            self.config.model,
            input_tokens=result.input_tokens or self.forecast_config.est_input_tokens,
            output_tokens=result.output_tokens or self.forecast_config.est_output_tokens,
            config=self.forecast_config,
            pricing=self.pricing,
        )
        return result


class NoOpForecastProvider:
    """Provider that always returns an error. Used when no API key is configured."""

    def __init__(self, reason: str = "no_api_key", model: str = "noop") -> None:
        self._reason = reason
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def forecast_one(self, request: ForecastRequest) -> ForecastResult:
        return ForecastResult(
            market_id=request.market_id,
            model=self._model,
            error=self._reason,
            edge_source="forecast_unavailable",
        )

    def close(self) -> None:
        return None


def _resolve_env(model: str, env: dict[str, str]) -> ProviderConfig | None:
    """Resolve API key + base URL from the environment for ``model``.

    Returns ``None`` if no compatible key is configured. We never print
    the key here; only the resolved base URL.
    """
    provider_prefix, _ = _classify_model(model)
    if provider_prefix == "openrouter":
        key = env.get("OPENROUTER_API_KEY")
        if not key:
            return None
        base = env.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE
    else:
        key = env.get("OPENAI_API_KEY") or env.get("OPENROUTER_API_KEY")
        if not key:
            return None
        base = env.get("OPENAI_BASE_URL") or env.get("OPENROUTER_BASE_URL") or DEFAULT_OPENAI_BASE
    _, model_name = _classify_model(model)
    tier = _model_tier_from_name(model_name)
    return ProviderConfig(model=model, api_key=key, base_url=base, model_tier=tier)


def build_default_provider(
    env: dict[str, str] | None = None,
    *,
    forecast_config: ForecastConfig | None = None,
) -> ForecastProvider:
    """Build a forecast provider from environment variables.

    If no compatible key is configured, returns :class:`NoOpForecastProvider`
    so the loop can still complete the tick and emit ``forecast_blocked``
    audit rows.
    """
    env = env if env is not None else dict(os.environ)
    fc = forecast_config or get_default_config()
    model = env.get("KALIBRE_FORECAST_MODEL") or fc.default_model
    cfg = _resolve_env(model, env)
    if cfg is None:
        logger.warning(
            "no forecast API key found; using NoOpForecastProvider for model %s",
            model,
        )
        return NoOpForecastProvider(reason="no_api_key", model=model)
    return HTTPForecastProvider(cfg, forecast_config=fc)
