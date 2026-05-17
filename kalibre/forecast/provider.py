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
import re
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


# Phase 6B: regex matches bare URLs in model output (fallback if the
# provider omits structured annotations). Deliberately conservative -- it
# does not try to dereference relative URLs or unicode hosts.
_URL_RE = re.compile(r"https?://[\w\-.]+(?:/[\w\-./?%&=~#:+]*)?", re.IGNORECASE)


def _extract_cited_urls(annotations: list, content: str) -> list[str]:
    """Pull cited URLs from OpenRouter response annotations + free text.

    OpenRouter's web-search tool returns annotation entries like
    ``{"type":"url_citation","url_citation":{"url":...,"title":...,
    "content":...,"start_index":...}}`` on the assistant message. Some
    engines also embed the URL inline in the rationale; the regex
    fallback catches those.
    """
    urls: list[str] = []
    seen: set[str] = set()
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            ann_type = str(ann.get("type") or "").lower()
            payload = ann.get("url_citation") if ann_type == "url_citation" else ann
            if not isinstance(payload, dict):
                continue
            url = payload.get("url") or payload.get("uri") or payload.get("href")
            if isinstance(url, str) and url not in seen:
                seen.add(url)
                urls.append(url)
    if isinstance(content, str):
        for m in _URL_RE.finditer(content):
            u = m.group(0).rstrip(").,;:'\"]")
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


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
    # Phase 6B: OpenRouter native web-search tool. When ``web_search_enabled``
    # is True the request body adds ``tools: [{"type":"openrouter:web_search",
    # "parameters": {...}}]``. The model decides whether to call the tool;
    # OpenRouter runs the search server-side and the response carries
    # ``usage.server_tool_use.web_search_requests`` + url_citation
    # annotations.
    web_search_enabled: bool = False
    web_search_engine: str = "exa"
    web_search_max_results: int = 4
    web_search_max_total_results: int = 12
    web_search_context_size: str = "low"
    web_search_allowed_domains: tuple[str, ...] = ()
    web_search_excluded_domains: tuple[str, ...] = ("reddit.com",)
    # Per-search unit cost (USD). OpenRouter docs quote ~$0.005 for
    # exa/parallel. The provider adds ``requests * cost`` to api_cost_usd
    # so spend_log stays reconciled.
    web_search_unit_cost_usd: float = 0.005
    # When web search is active the model spends many completion tokens
    # narrating its chain of thought over the retrieved evidence. The
    # default ``max_tokens=600`` (used for evidence-free forecasts) gets
    # truncated before the JSON line. 800 is enough headroom for the
    # JSON-first directive in the Phase 6C prompt; previously 1500.
    web_search_max_tokens: int = 800


class ForecastProvider(Protocol):
    """Single-market forecast provider. Errors live in ``ForecastResult``."""

    @property
    def model(self) -> str: ...

    def forecast_one(
        self,
        request: ForecastRequest,
        *,
        web_search_enabled_override: bool | None = None,
    ) -> ForecastResult: ...


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

    def forecast_one(
        self,
        request: ForecastRequest,
        *,
        web_search_enabled_override: bool | None = None,
    ) -> ForecastResult:
        # Phase 6C repair: caller can force web search off for a single
        # call (used by the runner when daily spend cap / min-universe /
        # deadline blocks search). The provider's config is unchanged.
        web_search_enabled = self.config.web_search_enabled
        if web_search_enabled_override is not None:
            web_search_enabled = bool(web_search_enabled_override)
        result = ForecastResult(
            market_id=request.market_id,
            model=self.config.model,
            model_tier=self.config.model_tier,
            edge_source=self.config.edge_source,
        )
        _, model_name = _classify_model(self.config.model)
        # Phase 6B: web-search responses include the retrieved evidence
        # in the prompt context and the model spends many completion
        # tokens narrating before emitting JSON. Use the larger cap.
        max_tokens = (
            self.config.web_search_max_tokens
            if web_search_enabled
            else self.config.max_tokens
        )
        body: dict[str, Any] = {
            "model": model_name,
            "messages": build_prompt(request),
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
        }
        # Phase 6B: OpenRouter web-search tool. Note: we drop
        # ``response_format=json_object`` when tools are present because
        # not all engines accept both; the halawi parser already extracts
        # JSON from prose so we lose nothing.
        if web_search_enabled:
            params: dict[str, Any] = {
                "engine": self.config.web_search_engine,
                "max_results": self.config.web_search_max_results,
                "max_total_results": self.config.web_search_max_total_results,
                "search_context_size": self.config.web_search_context_size,
            }
            if self.config.web_search_excluded_domains:
                params["excluded_domains"] = list(self.config.web_search_excluded_domains)
            if self.config.web_search_allowed_domains:
                params["allowed_domains"] = list(self.config.web_search_allowed_domains)
            body["tools"] = [{
                "type": "openrouter:web_search",
                "parameters": params,
            }]
        else:
            body["response_format"] = {"type": "json_object"}
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
            message = choice.get("message") or {}
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            result.error = f"unexpected_response_shape: {exc}"
            return result
        # Phase 6C forecast-repair: extract usage / citations / cost BEFORE
        # the parse attempt so a parse failure still carries the spend
        # signal and any evidence the model retrieved. Without this the
        # spend_log under-reports actual upstream costs whenever the
        # model emits prose-only or truncated JSON.
        usage = payload.get("usage") or {}
        result.input_tokens = int(usage.get("prompt_tokens") or 0)
        result.output_tokens = int(usage.get("completion_tokens") or 0)
        result.cached_tokens = int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )
        server_tool_use = usage.get("server_tool_use") or {}
        reported_requests = int(server_tool_use.get("web_search_requests") or 0)
        annotations = message.get("annotations") or []
        cited_urls = _extract_cited_urls(annotations, content)
        if cited_urls:
            result.cited_urls = tuple(cited_urls)
        if reported_requests > 0:
            result.web_search_requests = reported_requests
        elif web_search_enabled and cited_urls:
            denom = max(1, int(self.config.web_search_max_results or 1))
            est = max(1, (len(cited_urls) + denom - 1) // denom)
            result.web_search_requests = min(
                est, max(1, int(self.config.web_search_max_total_results or est)),
            )
        else:
            result.web_search_requests = 0
        token_cost = estimate_cost_usd(
            self.config.model,
            input_tokens=result.input_tokens or self.forecast_config.est_input_tokens,
            output_tokens=result.output_tokens or self.forecast_config.est_output_tokens,
            config=self.forecast_config,
            pricing=self.pricing,
        )
        search_cost = result.web_search_requests * float(
            self.config.web_search_unit_cost_usd
        )
        reported_total = usage.get("cost")
        try:
            reported_total = float(reported_total) if reported_total is not None else None
        except (TypeError, ValueError):
            reported_total = None
        modeled_total = token_cost + search_cost
        if reported_total is not None and reported_total > modeled_total:
            result.api_cost_usd = float(reported_total)
        else:
            result.api_cost_usd = float(modeled_total)
        try:
            parsed = parse_forecast_json(content)
        except ForecastParseError as exc:
            # Phase 6C forecast-repair: parse failure does not erase the
            # API spend or the cited evidence. Surface a short preview of
            # the model output so post-mortem can see what came back.
            preview = (content or "").strip().replace("\n", " ")[:200]
            result.error = f"parse_error: {exc}"
            result.rationale = (
                f"parse_error preview: {preview}" if preview else f"parse_error: {exc}"
            )
            return result
        result.p_raw = float(parsed["p_yes"])
        result.p_mean = result.p_raw  # callers calibrate later
        result.confidence = parsed.get("confidence")
        result.rationale = parsed.get("rationale") or ""
        # Phase 6B: pull out the expanded JSON fields (parser already
        # extracted them in a backwards-compatible way).
        if isinstance(parsed.get("category"), str):
            result.category = parsed["category"]
        if isinstance(parsed.get("evidence_quality"), (int, float)):
            result.evidence_quality = max(0.0, min(1.0, float(parsed["evidence_quality"])))
        if isinstance(parsed.get("stale_evidence"), bool):
            result.stale_evidence = bool(parsed["stale_evidence"])
        key_drivers = parsed.get("key_drivers")
        parsed_source_ids: list[str] = []
        if isinstance(key_drivers, list) and key_drivers:
            for driver in key_drivers:
                if not isinstance(driver, dict):
                    continue
                ids = driver.get("source_ids") or []
                if isinstance(ids, list):
                    for sid in ids:
                        if isinstance(sid, str):
                            parsed_source_ids.append(sid)
            import json as _json
            result.key_drivers_json = _json.dumps(key_drivers, default=str)
        if parsed_source_ids:
            result.source_ids = tuple(dict.fromkeys(parsed_source_ids))
        return result


class NoOpForecastProvider:
    """Provider that always returns an error. Used when no API key is configured."""

    def __init__(self, reason: str = "no_api_key", model: str = "noop") -> None:
        self._reason = reason
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def forecast_one(
        self,
        request: ForecastRequest,
        *,
        web_search_enabled_override: bool | None = None,
    ) -> ForecastResult:
        # Override is accepted for protocol compatibility; the no-op
        # provider doesn't issue HTTP so the flag has no effect here.
        del web_search_enabled_override
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
    web_search_overrides: dict[str, Any] | None = None,
) -> ForecastProvider:
    """Build a forecast provider from environment variables.

    If no compatible key is configured, returns :class:`NoOpForecastProvider`
    so the loop can still complete the tick and emit ``forecast_blocked``
    audit rows.

    Phase 6B: ``web_search_overrides`` is a flat dict of ProviderConfig
    field overrides (``web_search_enabled``, ``web_search_engine``, etc.).
    The kalibre.forecast.web_search loader passes this in based on env.
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
    if web_search_overrides:
        cfg = _apply_web_search_overrides(cfg, web_search_overrides)
    return HTTPForecastProvider(cfg, forecast_config=fc)


def _apply_web_search_overrides(
    cfg: ProviderConfig, overrides: dict[str, Any],
) -> ProviderConfig:
    """Return a copy of ``cfg`` with the listed Phase 6B fields overridden.

    Frozen dataclass + dataclasses.replace semantics keep this immutable
    and avoid the long positional ProviderConfig constructor call.
    """
    from dataclasses import replace as _replace
    allowed = {
        "web_search_enabled", "web_search_engine", "web_search_max_results",
        "web_search_max_total_results", "web_search_context_size",
        "web_search_allowed_domains", "web_search_excluded_domains",
        "web_search_unit_cost_usd",
    }
    payload = {k: v for k, v in overrides.items() if k in allowed}
    if not payload:
        return cfg
    return _replace(cfg, **payload)


# --- Phase 6: Opus escalation provider builder ----------------------------


DEFAULT_OPUS_MODEL = "openrouter:anthropic/claude-opus-4"


def build_opus_provider(
    env: dict[str, str] | None = None,
    *,
    forecast_config: ForecastConfig | None = None,
    model: str | None = None,
    edge_source: str = "opus_forecast",
    web_search_overrides: dict[str, Any] | None = None,
) -> ForecastProvider:
    """Build a Phase 6 Opus escalation provider.

    Mirrors :func:`build_default_provider` but pins the model to Opus 4 by
    default and tags the result with ``model_tier="opus"`` so audit rows
    can split Sonnet vs Opus calls cleanly. Returns
    :class:`NoOpForecastProvider` if no API key is available.
    """
    env = env if env is not None else dict(os.environ)
    fc = forecast_config or get_default_config()
    model = (
        model
        or env.get("KALIBRE_OPUS_FORECAST_MODEL")
        or DEFAULT_OPUS_MODEL
    )
    cfg = _resolve_env(model, env)
    if cfg is None:
        logger.warning(
            "no Opus API key found; using NoOpForecastProvider for model %s",
            model,
        )
        return NoOpForecastProvider(reason="no_api_key", model=model)
    # Force tier=opus and edge_source regardless of whether the model name
    # heuristic in `_resolve_env` matched.
    from dataclasses import replace as _replace
    cfg = _replace(cfg, edge_source=edge_source, model_tier="opus")
    if web_search_overrides:
        cfg = _apply_web_search_overrides(cfg, web_search_overrides)
    return HTTPForecastProvider(cfg, forecast_config=fc)
