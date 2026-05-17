"""Forecast cache (memory + SQLite implementations).

Cache key is a stable hash of:

- ``market_id``
- ``question`` (canonicalized whitespace)
- ``resolution_time`` ISO
- a quote signature ``(round(bid, 3), round(ask, 3), round(volume, -1))``
- ``model``

TTL defaults to 6h. Anything older than ``ttl_sec`` is treated as a miss
even if it's still on disk.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Protocol

from kalibre.forecast.types import ForecastRequest, ForecastResult


DEFAULT_TTL_SEC = 6 * 3600


def cache_key(request: ForecastRequest, model: str) -> str:
    """Stable hash of the inputs that should yield the same forecast."""
    question = " ".join((request.question or "").split())
    resolution = (
        request.resolution_time.isoformat() if request.resolution_time else ""
    )
    bid = round(request.best_bid, 3) if request.best_bid is not None else None
    ask = round(request.best_ask, 3) if request.best_ask is not None else None
    volume = (
        round(request.volume_24h, -1) if request.volume_24h is not None else None
    )
    raw = json.dumps(
        {
            "market_id": request.market_id,
            "question": question,
            "resolution": resolution,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ForecastCache(Protocol):
    def get(self, key: str, *, ttl_sec: int = DEFAULT_TTL_SEC) -> ForecastResult | None: ...
    def put(self, key: str, result: ForecastResult) -> None: ...


# --- in-memory cache --------------------------------------------------------


class MemoryForecastCache:
    """Simple dict-backed cache used by tests."""

    def __init__(self, *, time_fn: Any = time.time) -> None:
        self._store: dict[str, tuple[float, ForecastResult]] = {}
        self._time_fn = time_fn

    def get(self, key: str, *, ttl_sec: int = DEFAULT_TTL_SEC) -> ForecastResult | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_ts, result = entry
        if self._time_fn() - stored_ts > ttl_sec:
            return None
        # Mark cache_hit so callers can audit it correctly.
        return _clone_as_cache_hit(result)

    def put(self, key: str, result: ForecastResult) -> None:
        # Never cache errors -- a failed call should be retried.
        if result.error is not None or result.p_raw is None:
            return
        self._store[key] = (self._time_fn(), result)


# --- SQLite cache -----------------------------------------------------------


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forecast_cache (
    cache_key TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    model TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecast_cache_created
ON forecast_cache(created_at);
"""


class SQLiteForecastCache:
    """File-backed cache that piggy-backs on the existing state DB connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.executescript(SCHEMA_SQL)

    def get(self, key: str, *, ttl_sec: int = DEFAULT_TTL_SEC) -> ForecastResult | None:
        row = self.conn.execute(
            "SELECT payload_json, created_at FROM forecast_cache WHERE cache_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload_json, created_at = row
        try:
            created_dt = datetime.fromisoformat(created_at)
        except ValueError:
            return None
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=UTC)
        age = (datetime.now(tz=UTC) - created_dt).total_seconds()
        if age > ttl_sec:
            return None
        try:
            data = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return _clone_as_cache_hit(_result_from_dict(data))

    def put(self, key: str, result: ForecastResult) -> None:
        if result.error is not None or result.p_raw is None:
            return
        payload = json.dumps(asdict(result), default=str)
        now_iso = datetime.now(tz=UTC).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO forecast_cache(cache_key, market_id, model, payload_json, created_at) "
            "VALUES (?,?,?,?,?)",
            (key, result.market_id, result.model, payload, now_iso),
        )


# --- helpers ----------------------------------------------------------------


def _clone_as_cache_hit(result: ForecastResult) -> ForecastResult:
    clone = ForecastResult(market_id=result.market_id)
    for field, value in asdict(result).items():
        setattr(clone, field, value)
    clone.cache_hit = True
    # Cost should not be re-charged on a hit.
    clone.api_cost_usd = 0.0
    clone.error = None  # paranoia: should already be None
    return clone


def _result_from_dict(data: dict[str, Any]) -> ForecastResult:
    return ForecastResult(
        market_id=str(data.get("market_id", "")),
        p_raw=data.get("p_raw"),
        p_mean=data.get("p_mean"),
        sigma_p=float(data.get("sigma_p", 0.10)),
        confidence=data.get("confidence"),
        model_tier=str(data.get("model_tier", "external")),
        edge_source=str(data.get("edge_source", "sonnet_forecast")),
        rationale=str(data.get("rationale", "")),
        api_cost_usd=float(data.get("api_cost_usd", 0.0)),
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        cached_tokens=int(data.get("cached_tokens", 0) or 0),
        cache_hit=bool(data.get("cache_hit", False)),
        error=data.get("error"),
        model=str(data.get("model", "")),
    )
