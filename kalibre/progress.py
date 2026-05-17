"""Per-experiment progress artifacts.

For every run we materialize a directory
``experiments/<slug>__<experiment_id>/`` and keep four files current:

- ``out.log``: JSONL structured events.
- ``summary.csv``: one row per processed tick using the columns mandated
  by the T0 spec.
- ``live_progress.md``: human-readable progress with ETA, percent, and
  text progress bar.
- ``state.sqlite3``: SQLite WAL audit DB (owned by :mod:`kalibre.state`).
"""

from __future__ import annotations

import contextlib
import csv
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_ts",
    "experiment_id",
    "slug",
    "participant_idx",
    "tick_id",
    "candidate_set_id",
    "status",
    "candidates_loaded",
    "intents_submitted",
    "accepted",
    "rejected",
    "fills",
    "cash",
    "equity",
    "total_pnl",
    "error_code",
    "error_detail",
    "elapsed_sec",
    "deadline_slip_ms",
    "fallback_triggered",
)


def utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _render_bar(pct: float, width: int = 40) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "#" * filled + "-" * (width - filled)


class ExperimentDir:
    """Owns the per-experiment artifacts directory."""

    def __init__(self, root: Path, slug: str, experiment_id: str) -> None:
        self.root = Path(root)
        self.slug = slug
        self.experiment_id = experiment_id
        self.dir = self.root / f"{slug}__{experiment_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.out_log_path = self.dir / "out.log"
        self.summary_csv_path = self.dir / "summary.csv"
        self.live_progress_path = self.dir / "live_progress.md"
        self.state_db_path = self.dir / "state.sqlite3"
        self._log_lock = threading.Lock()
        self._csv_lock = threading.Lock()
        self._init_summary()

    def _init_summary(self) -> None:
        if self.summary_csv_path.exists() and self.summary_csv_path.stat().st_size > 0:
            return
        with self.summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()

    # --- log -----------------------------------------------------------------

    def log_event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": utcnow_iso(),
            "event": event,
            "experiment_id": self.experiment_id,
            "slug": self.slug,
        }
        record.update(fields)
        line = json.dumps(record, default=_json_default)
        with self._log_lock, self.out_log_path.open("a") as f:
            f.write(line + "\n")

    # --- summary csv ---------------------------------------------------------

    def append_summary_row(self, row: dict[str, Any]) -> None:
        filled: dict[str, Any] = {}
        for col in SUMMARY_COLUMNS:
            value = row.get(col, "")
            if isinstance(value, bool):
                value = int(value)
            filled[col] = "" if value is None else value
        with self._csv_lock, self.summary_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writerow(filled)

    # --- live progress -------------------------------------------------------

    def update_live_progress(
        self,
        *,
        completed: int,
        target: int,
        started_at: datetime,
        last_tick_id: str | None,
        last_status: str | None,
    ) -> None:
        safe_target = max(1, int(target))
        clamped = max(0, min(int(completed), safe_target))
        pct = 100.0 * clamped / safe_target
        bar = _render_bar(pct)
        now = datetime.now(tz=UTC)
        elapsed = max(0.0, (now - started_at).total_seconds())
        if clamped >= safe_target:
            eta_label = "complete"
            remaining_label = "0s"
        elif clamped > 0:
            per_tick = elapsed / clamped
            remaining = per_tick * (safe_target - clamped)
            eta_label = (now + timedelta(seconds=remaining)).isoformat(
                timespec="seconds"
            )
            remaining_label = f"{remaining:,.1f}s"
        else:
            eta_label = "unknown"
            remaining_label = "unknown"
        body = (
            f"# Progress - {self.slug}\n\n"
            f"- experiment_id: `{self.experiment_id}`\n"
            f"- target ticks: {safe_target}\n"
            f"- completed ticks: {clamped}\n"
            f"- percent complete: {pct:.1f}%\n"
            f"- ETA: {eta_label}\n"
            f"- remaining (estimated): {remaining_label}\n"
            f"- last tick_id: {last_tick_id or '-'}\n"
            f"- last status: {last_status or '-'}\n"
            f"- updated: {now.isoformat(timespec='seconds')}\n"
            f"- started: {started_at.isoformat(timespec='seconds')}\n\n"
            f"```\n[{bar}] {pct:5.1f}%\n```\n"
        )
        tmp = self.live_progress_path.with_suffix(".md.tmp")
        try:
            tmp.write_text(body)
            tmp.replace(self.live_progress_path)
        except OSError:
            # Fallback: write directly. Atomicity loss is acceptable for
            # a progress hint.
            with contextlib.suppress(OSError):
                self.live_progress_path.write_text(body)
