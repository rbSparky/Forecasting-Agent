"""SQLite WAL state store with checkpoints and corrupt-DB recovery.

T0 schema is intentionally minimal: only what the survival shell needs
to audit ticks, intents, and (future) spend. Richer tables from
``AGENTS.md`` §C.2 land in later tiers.

The store:

- Opens SQLite in WAL mode with NORMAL sync.
- Runs ``PRAGMA integrity_check`` on every open and, if the file is
  corrupt, swaps in the most recent checkpoint before reconnecting.
- Exposes ``checkpoint()`` to take a full online backup into
  ``checkpoints/checkpoint_<ts>.sqlite3``.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tick_audit (
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    candidate_set_id TEXT,
    status TEXT NOT NULL,
    candidates_loaded INTEGER NOT NULL DEFAULT 0,
    intents_submitted INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    rejected INTEGER NOT NULL DEFAULT 0,
    fills INTEGER NOT NULL DEFAULT 0,
    cash TEXT,
    equity TEXT,
    total_pnl TEXT,
    error_code TEXT,
    error_detail TEXT,
    elapsed_sec REAL,
    deadline_slip_ms INTEGER,
    fallback_triggered INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tick_ts, experiment_id, participant_idx)
);

CREATE TABLE IF NOT EXISTS intents (
    intent_id TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT,
    action TEXT,
    shares TEXT,
    edge_source TEXT NOT NULL,
    version TEXT NOT NULL,
    audit_json TEXT NOT NULL,
    fill_status TEXT,
    PRIMARY KEY (tick_ts, experiment_id, participant_idx, intent_id)
);

CREATE TABLE IF NOT EXISTS spend_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    day_key TEXT NOT NULL,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    purpose TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    edge_source TEXT NOT NULL,
    version TEXT NOT NULL,
    p_mean REAL,
    p_eff REAL,
    sigma_p REAL,
    side TEXT,
    action TEXT,
    size_usd REAL,
    shares REAL,
    score REAL,
    fill_prob REAL,
    decision TEXT NOT NULL,
    reject_reason TEXT,
    audit_json TEXT NOT NULL,
    PRIMARY KEY (tick_ts, experiment_id, participant_idx, proposal_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    model TEXT,
    model_tier TEXT,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    p_raw REAL,
    p_cal REAL,
    sigma_p REAL,
    p_market REAL,
    p_blend REAL,
    blend_reason TEXT,
    api_cost_usd REAL NOT NULL DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    rationale TEXT,
    error TEXT,
    edge_source TEXT NOT NULL,
    decision TEXT NOT NULL,
    audit_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_market_tick
ON forecasts(market_id, tick_ts);
CREATE INDEX IF NOT EXISTS idx_forecasts_tick
ON forecasts(tick_ts);

CREATE TABLE IF NOT EXISTS calibration_model (
    bucket_key TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    params_json TEXT NOT NULL,
    n_samples INTEGER NOT NULL DEFAULT 0,
    brier REAL,
    log_loss REAL,
    last_fit_ts TEXT NOT NULL
);

-- Phase 4A: per-tick raw quote snapshot for every loaded candidate. Used
-- to compute the A.11 quote-history features the following tick.
CREATE TABLE IF NOT EXISTS quote_history (
    quote_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    mid REAL,
    spread REAL,
    volume_24h REAL,
    quote_ts TEXT,
    quote_age_sec REAL,
    source TEXT,
    topic TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(market_id, tick_ts, experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_quote_history_market_tick
ON quote_history(market_id, tick_ts);

-- Phase 4A: A.11 computed features per (market, tick).
CREATE TABLE IF NOT EXISTS quote_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    mid_return_1_tick REAL,
    mid_return_4_tick REAL,
    mid_return_8_tick REAL,
    spread_current REAL,
    spread_change_4_tick REAL,
    quote_stability REAL,
    volatility_8_tick REAL,
    time_since_seen REAL,
    qh_warmup INTEGER NOT NULL DEFAULT 0,
    market_is_smart INTEGER NOT NULL DEFAULT 0,
    qh_confidence_modifier REAL,
    deference_extra_shrinkage REAL,
    exit_pressure REAL,
    features_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(market_id, tick_ts, experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_quote_features_market_tick
ON quote_features(market_id, tick_ts);

-- Phase 4A: shadow-variant audit. Never used to size or submit real trades.
CREATE TABLE IF NOT EXISTS shadow_proposals (
    shadow_id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_name TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    edge_source TEXT NOT NULL,
    p_mean REAL,
    sigma_p REAL,
    side TEXT,
    score REAL,
    decision TEXT NOT NULL,
    reject_reason TEXT,
    audit_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_tick
ON shadow_proposals(tick_ts, variant_name);
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_market
ON shadow_proposals(market_id, variant_name);

-- Phase 4A: category-level priors + stop-loss state for A.6 longshot.
CREATE TABLE IF NOT EXISTS category_priors (
    category TEXT NOT NULL,
    extreme_side TEXT NOT NULL,
    alpha_adjustment REAL NOT NULL,
    n_resolved_last20 INTEGER NOT NULL DEFAULT 0,
    n_wins_last20 INTEGER NOT NULL DEFAULT 0,
    win_rate_last20 REAL,
    stop_loss_until_ts TEXT,
    updated_at TEXT,
    PRIMARY KEY (category, extreme_side)
);

-- Phase 4A: per-trade longshot outcomes, used to fill the 20-trade window.
CREATE TABLE IF NOT EXISTS longshot_outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    extreme_side TEXT NOT NULL,
    market_id TEXT NOT NULL,
    result INTEGER NOT NULL,  -- 1 = win, 0 = loss
    resolved_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_longshot_outcomes_category
ON longshot_outcomes(category, extreme_side, resolved_at);

-- Phase 5A: A.9 fill / rejection telemetry. One row per submitted intent.
-- Whatever the SDK returns (FillData / RejectionData) is reconciled here so
-- a downstream report can compute realized fill rate + slippage. Missing
-- SDK fields are stored as NULL; never raises.
CREATE TABLE IF NOT EXISTS fills (
    fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id TEXT NOT NULL,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    participant_idx INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT,
    action TEXT,
    quoted_bid_at_decision REAL,
    quoted_ask_at_decision REAL,
    mid_at_decision REAL,
    quote_age_sec_at_decision REAL,
    submitted_shares REAL,
    submitted_price_implied REAL,
    edge_source TEXT,
    expected_fill_prob REAL,
    fill_status TEXT NOT NULL,
    filled_shares REAL,
    filled_price REAL,
    fill_delay_ms INTEGER,
    rejection_reason TEXT,
    slippage_bps REAL,
    post_fill_mid REAL,
    audit_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (intent_id, tick_ts, experiment_id, participant_idx)
);
CREATE INDEX IF NOT EXISTS idx_fills_tick ON fills(tick_ts, experiment_id);
CREATE INDEX IF NOT EXISTS idx_fills_market ON fills(market_id, tick_ts);
CREATE INDEX IF NOT EXISTS idx_fills_edge_source ON fills(edge_source);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Phase 6B: OpenRouter web-search auditing. The model decides when to
-- search; the response carries usage.server_tool_use.web_search_requests
-- plus url_citation annotations. We persist one row per cited URL into
-- web_search_results and one row per forecast bundle into
-- evidence_bundles. ``web_search_queries`` is a per-forecast aggregate
-- (we don't see the individual queries from OpenRouter).
CREATE TABLE IF NOT EXISTS web_search_queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    engine TEXT,
    requests_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wsq_tick ON web_search_queries(tick_ts);
CREATE INDEX IF NOT EXISTS idx_wsq_market ON web_search_queries(market_id, tick_ts);

CREATE TABLE IF NOT EXISTS web_search_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    url TEXT NOT NULL,
    domain TEXT,
    source_tier TEXT,
    title TEXT,
    snippet TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wsr_tick ON web_search_results(tick_ts);
CREATE INDEX IF NOT EXISTS idx_wsr_market ON web_search_results(market_id, tick_ts);
CREATE INDEX IF NOT EXISTS idx_wsr_domain ON web_search_results(domain);

CREATE TABLE IF NOT EXISTS evidence_bundles (
    bundle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_ts TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    evidence_hash TEXT,
    urls_json TEXT,
    evidence_quality REAL,
    stale_evidence INTEGER,
    key_drivers_json TEXT,
    web_search_requests INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eb_tick ON evidence_bundles(tick_ts);
CREATE INDEX IF NOT EXISTS idx_eb_market ON evidence_bundles(market_id, tick_ts);
"""


class StateStore:
    """Thin SQLite wrapper that enforces WAL + integrity-on-open."""

    SCHEMA_VERSION = "1"

    def __init__(
        self,
        db_path: Path,
        *,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None
            else self.db_path.parent / "checkpoints"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self.db_path.exists() and not self._integrity_ok(self.db_path):
            logger.warning(
                "state.sqlite3 integrity check failed; restoring from latest checkpoint",
            )
            self._restore_from_latest_checkpoint()
        self._conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._configure_connection()
        self._ensure_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is None:
            return
        with contextlib.suppress(Exception):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with contextlib.suppress(Exception):
            self._conn.close()
        self._conn = None

    def __enter__(self) -> "StateStore":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- checkpoint / recovery ---------------------------------------------

    def checkpoint(self) -> Path:
        """Take a full online backup. Returns the new checkpoint path."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
        out = self.checkpoint_dir / f"checkpoint_{ts}.sqlite3"
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")
        dst = sqlite3.connect(str(out))
        try:
            self._conn.backup(dst)
        finally:
            dst.close()
        logger.info("state checkpoint written: %s", out)
        return out

    @staticmethod
    def _integrity_ok(path: Path) -> bool:
        try:
            conn = sqlite3.connect(str(path))
        except sqlite3.DatabaseError as exc:
            logger.warning("cannot open %s for integrity check: %s", path, exc)
            return False
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            logger.warning("integrity_check raised on %s: %s", path, exc)
            return False
        finally:
            with contextlib.suppress(Exception):
                conn.close()
        return bool(row and str(row[0]).lower() == "ok")

    def _restore_from_latest_checkpoint(self) -> None:
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*.sqlite3"))
        # Filter only checkpoints that themselves pass integrity_check.
        ckpts = [c for c in ckpts if self._integrity_ok(c)]
        corrupt_dest = self.db_path.with_suffix(self.db_path.suffix + ".corrupt")
        if not ckpts:
            with contextlib.suppress(FileNotFoundError):
                self.db_path.replace(corrupt_dest)
            # Also clear any stray -wal/-shm sidecars.
            for sidecar in (
                self.db_path.with_suffix(self.db_path.suffix + "-wal"),
                self.db_path.with_suffix(self.db_path.suffix + "-shm"),
            ):
                with contextlib.suppress(FileNotFoundError):
                    sidecar.unlink()
            logger.warning(
                "no checkpoint available; moved corrupt db to %s and starting fresh",
                corrupt_dest,
            )
            return
        latest = ckpts[-1]
        with contextlib.suppress(FileNotFoundError):
            self.db_path.replace(corrupt_dest)
        for sidecar in (
            self.db_path.with_suffix(self.db_path.suffix + "-wal"),
            self.db_path.with_suffix(self.db_path.suffix + "-shm"),
        ):
            with contextlib.suppress(FileNotFoundError):
                sidecar.unlink()
        shutil.copy2(latest, self.db_path)
        logger.info("restored state from checkpoint %s", latest.name)

    # --- internals ----------------------------------------------------------

    def _configure_connection(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
        finally:
            cur.close()

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(SCHEMA_SQL)
        self._apply_additive_migrations()
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (self.SCHEMA_VERSION,),
        )

    def _apply_additive_migrations(self) -> None:
        """Bring older state DBs up to the current schema additively.

        ``CREATE TABLE IF NOT EXISTS`` doesn't add new columns to an
        existing table, so we ``ALTER TABLE ... ADD COLUMN`` here.
        Errors mean the column is already present; we swallow them.
        """
        assert self._conn is not None
        for ddl in (
            # Phase 3B blend columns on forecasts.
            "ALTER TABLE forecasts ADD COLUMN p_market REAL",
            "ALTER TABLE forecasts ADD COLUMN p_blend REAL",
            "ALTER TABLE forecasts ADD COLUMN blend_reason TEXT",
            # Phase 4A: backfill columns onto pre-existing category_priors
            # rows that older runs may have created with a slimmer schema.
            "ALTER TABLE category_priors ADD COLUMN n_resolved_last20 INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE category_priors ADD COLUMN n_wins_last20 INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE category_priors ADD COLUMN win_rate_last20 REAL",
            "ALTER TABLE category_priors ADD COLUMN stop_loss_until_ts TEXT",
            "ALTER TABLE category_priors ADD COLUMN updated_at TEXT",
        ):
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                # Column already exists. Safe to ignore on the additive path.
                pass

    # --- audit helpers ------------------------------------------------------

    def record_tick(self, row: dict[str, Any]) -> None:
        conn = self.connect()
        cols = list(row.keys())
        placeholders = ", ".join("?" * len(cols))
        conn.execute(
            f"INSERT OR REPLACE INTO tick_audit({', '.join(cols)}) VALUES ({placeholders})",
            tuple(row[c] for c in cols),
        )

    def record_intents(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO intents({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_quote_history(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO quote_history({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_quote_features(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO quote_features({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_shadow_proposals(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT INTO shadow_proposals({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_fills(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist Phase 5A fill audit rows (one per submitted intent)."""
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO fills({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_longshot_outcome(
        self,
        *,
        category: str,
        extreme_side: str,
        market_id: str,
        result: int,
        resolved_at: datetime | None = None,
    ) -> None:
        conn = self.connect()
        ts = (resolved_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
        conn.execute(
            "INSERT INTO longshot_outcomes(category, extreme_side, market_id, result, resolved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, extreme_side, market_id, int(result), ts),
        )

    def upsert_category_prior(
        self,
        *,
        category: str,
        extreme_side: str,
        alpha_adjustment: float,
        n_resolved_last20: int,
        n_wins_last20: int,
        win_rate_last20: float | None,
        stop_loss_until_ts: str | None,
    ) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT OR REPLACE INTO category_priors"
            "(category, extreme_side, alpha_adjustment, n_resolved_last20, n_wins_last20, "
            "win_rate_last20, stop_loss_until_ts, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                category, extreme_side, float(alpha_adjustment),
                int(n_resolved_last20), int(n_wins_last20),
                win_rate_last20, stop_loss_until_ts,
                datetime.now(tz=UTC).isoformat(),
            ),
        )

    def record_forecasts(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT INTO forecasts({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def upsert_calibrator(
        self,
        *,
        bucket_key: str,
        method: str,
        params_json: str,
        n_samples: int = 0,
        brier: float | None = None,
        log_loss: float | None = None,
        last_fit_ts: str | None = None,
    ) -> None:
        conn = self.connect()
        last_ts = last_fit_ts or datetime.now(tz=UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO calibration_model"
            "(bucket_key, method, params_json, n_samples, brier, log_loss, last_fit_ts) "
            "VALUES (?,?,?,?,?,?,?)",
            (bucket_key, method, params_json, int(n_samples), brier, log_loss, last_ts),
        )

    def record_proposals(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO proposals({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    # --- Phase 6B: web-search audit writers ----------------------------

    def record_web_search_results(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist one row per cited URL into ``web_search_results``.

        Each row dict carries: ``tick_ts``, ``experiment_id``, ``market_id``,
        ``url``, ``domain``, ``source_tier``, optional ``title``/``snippet``,
        and ``created_at``. Unknown columns are tolerated -- the writer
        materializes ``INSERT`` from whichever subset is present.
        """
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT INTO web_search_results({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_evidence_bundles(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist one row per forecast bundle into ``evidence_bundles``."""
        materialized = list(rows)
        if not materialized:
            return
        conn = self.connect()
        cols = list(materialized[0].keys())
        placeholders = ", ".join("?" * len(cols))
        conn.executemany(
            f"INSERT INTO evidence_bundles({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in materialized],
        )

    def record_web_search_query(
        self,
        *,
        tick_ts: str,
        experiment_id: str,
        market_id: str,
        engine: str | None,
        requests_count: int,
        when: datetime | None = None,
    ) -> None:
        """Persist a per-forecast aggregate row into ``web_search_queries``."""
        conn = self.connect()
        ts = (when or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
        conn.execute(
            "INSERT INTO web_search_queries(tick_ts, experiment_id, market_id, "
            "engine, requests_count, created_at) VALUES (?,?,?,?,?,?)",
            (tick_ts, experiment_id, market_id, engine, int(requests_count or 0), ts),
        )

    def record_spend(
        self,
        *,
        model: str,
        cost_usd: float,
        purpose: str | None = None,
        when: datetime | None = None,
    ) -> None:
        conn = self.connect()
        when = when or datetime.now(tz=UTC)
        conn.execute(
            "INSERT INTO spend_log(day_key, ts, model, cost_usd, purpose) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                when.astimezone(UTC).date().isoformat(),
                when.astimezone(UTC).isoformat(),
                model,
                float(cost_usd),
                purpose,
            ),
        )
