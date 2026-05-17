"""Phase 5B production preflight.

CLI::

    python -m kalibre.preflight <experiment_folder_or_state.sqlite3>
    python -m kalibre.preflight <path> --json
    python -m kalibre.preflight <path> --slug-hint <slug>

The preflight is read-only and never submits anything. It opens the
experiment's ``state.sqlite3`` and asserts that the loop's invariants
hold tightly enough to attempt a controlled live canary tick:

- DB integrity and required tables present;
- latest tick completed cleanly (no fallback, no deadline slip);
- spend under the daily-hard / total-budget caps;
- fill telemetry schema present;
- fill_report + shadow_report both importable + buildable on this DB;
- at least one accepted proposal exists across the experiment so the
  canary can ride a real edge rather than forcing a trade;
- ``KALIBRE_ENABLE_LIVE_TRADES`` and ``KALIBRE_LIVE_CANARY_MODE`` are set
  to 1 with the documented safety caps if the operator is preparing to
  fire the canary.

Output:

- ``PreflightReport.live_canary_eligible`` is the single boolean gate
  callers should consult.
- ``blocking_reasons`` lists every reason a live canary is currently
  blocked, in priority order.
- ``suggested_dry_run_command`` and ``suggested_live_canary_command``
  give operators exact next commands keyed off ``--slug-hint``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --- threshold constants ---------------------------------------------------


REQUIRED_TABLES = (
    "tick_audit", "proposals", "forecasts", "shadow_proposals",
    "intents", "fills", "spend_log",
)
DEFAULT_DEADLINE_SLIP_MS_MAX = 0
DEFAULT_DAILY_HARD_USD = 35.0
DEFAULT_TOTAL_BUDGET_USD = 500.0
LIVE_CANARY_MAX_INTENTS_CAP = 1
LIVE_CANARY_MAX_SIZE_USD_CAP = 50.0


# --- dataclasses -----------------------------------------------------------


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass
class PreflightReport:
    db_path: str
    checks: list[PreflightCheck] = field(default_factory=list)
    accepted_proposals_count: int = 0
    last_tick_ts: str | None = None
    last_tick_status: str | None = None
    last_tick_fallback_triggered: int | None = None
    last_tick_deadline_slip_ms: int | None = None
    spend_daily_usd: float = 0.0
    spend_total_usd: float = 0.0
    fills_count: int = 0
    fills_filled_count: int = 0
    live_canary_eligible: bool = False
    blocking_reasons: list[str] = field(default_factory=list)
    env_state: dict[str, str] = field(default_factory=dict)
    suggested_dry_run_command: str = ""
    suggested_live_canary_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- helpers ---------------------------------------------------------------


def resolve_db_path(path: str | Path) -> Path:
    """Accept either ``state.sqlite3`` or its experiment folder."""
    p = Path(path)
    if p.is_dir():
        candidate = p / "state.sqlite3"
        if not candidate.exists():
            raise FileNotFoundError(f"state.sqlite3 not found in {p}")
        return candidate
    return p


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return row[0]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(report: PreflightReport, name: str, passed: bool, detail: str, severity: str = "error") -> None:
    report.checks.append(PreflightCheck(name=name, passed=passed, detail=detail, severity=severity))


def _bool_env(env: dict[str, str], key: str) -> bool:
    return (env.get(key) or "").strip() == "1"


def _build_dry_run_command(slug: str) -> str:
    py = shlex.quote(sys.executable)
    return (
        "source env.sh && source .venv/bin/activate && "
        f"KALIBRE_STRATEGY_MODE=forecast_dry_run "
        f"KALIBRE_ENABLE_LIVE_TRADES=0 "
        f"PA_EXPERIMENT_SLUG={shlex.quote(slug)} "
        f"PA_MAX_TICKS=1 "
        f"{py} -m kalibre.loop"
    )


def _build_live_canary_command(slug: str) -> str:
    py = shlex.quote(sys.executable)
    return (
        "source env.sh && source .venv/bin/activate && "
        f"KALIBRE_STRATEGY_MODE=forecast_dry_run "
        f"KALIBRE_ENABLE_LIVE_TRADES=1 "
        f"KALIBRE_LIVE_CANARY_MODE=1 "
        f"KALIBRE_CANARY_MAX_INTENTS={LIVE_CANARY_MAX_INTENTS_CAP} "
        f"KALIBRE_CANARY_MAX_SIZE_USD={int(LIVE_CANARY_MAX_SIZE_USD_CAP)} "
        f"PA_EXPERIMENT_SLUG={shlex.quote(slug)} "
        f"PA_MAX_TICKS=1 "
        f"{py} -m kalibre.loop"
    )


# --- core runner -----------------------------------------------------------


def run_preflight(
    path: str | Path,
    *,
    env: dict[str, str] | None = None,
    slug_hint: str | None = None,
    daily_hard_usd: float = DEFAULT_DAILY_HARD_USD,
    total_budget_usd: float = DEFAULT_TOTAL_BUDGET_USD,
    deadline_slip_ms_max: int = DEFAULT_DEADLINE_SLIP_MS_MAX,
) -> PreflightReport:
    """Run every preflight check against ``path`` and return the report."""
    resolved = resolve_db_path(path)
    env = env if env is not None else dict(os.environ)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    slug_for_dryrun = slug_hint or f"kalibre-preflight-dryrun-{ts}"
    slug_for_canary = (slug_hint and f"{slug_hint}-canary") or f"kalibre-live-canary-{ts}"
    report = PreflightReport(
        db_path=str(resolved),
        suggested_dry_run_command=_build_dry_run_command(slug_for_dryrun),
        env_state={
            "KALIBRE_STRATEGY_MODE": env.get("KALIBRE_STRATEGY_MODE") or "<unset>",
            "KALIBRE_ENABLE_LIVE_TRADES": env.get("KALIBRE_ENABLE_LIVE_TRADES") or "<unset>",
            "KALIBRE_LIVE_CANARY_MODE": env.get("KALIBRE_LIVE_CANARY_MODE") or "<unset>",
            "KALIBRE_CANARY_MAX_INTENTS": env.get("KALIBRE_CANARY_MAX_INTENTS") or "<unset>",
            "KALIBRE_CANARY_MAX_SIZE_USD": env.get("KALIBRE_CANARY_MAX_SIZE_USD") or "<unset>",
            "PA_EXPERIMENT_SLUG": env.get("PA_EXPERIMENT_SLUG") or "<unset>",
        },
    )

    # --- DB integrity -------------------------------------------------------
    try:
        conn = sqlite3.connect(str(resolved))
    except sqlite3.DatabaseError as exc:
        _check(report, "db_open", False, f"cannot open {resolved}: {exc}")
        report.blocking_reasons.append("db_unreachable")
        return report

    try:
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            _check(report, "db_integrity", False, f"integrity_check raised: {exc}")
            report.blocking_reasons.append("db_corrupt")
            return report
        if not integrity or str(integrity[0]).lower() != "ok":
            _check(report, "db_integrity", False, f"integrity_check returned {integrity}")
            report.blocking_reasons.append("db_corrupt")
            return report
        _check(report, "db_integrity", True, "PRAGMA integrity_check=ok")

        # --- required tables -----------------------------------------------
        missing = [t for t in REQUIRED_TABLES if not _table_exists(conn, t)]
        if missing:
            _check(
                report, "required_tables", False,
                f"missing tables: {', '.join(missing)}",
            )
            report.blocking_reasons.append(f"missing_tables:{','.join(missing)}")
        else:
            _check(
                report, "required_tables", True,
                f"present: {', '.join(REQUIRED_TABLES)}",
            )

        # --- latest tick clean ---------------------------------------------
        latest_row = None
        if _table_exists(conn, "tick_audit"):
            latest_row = conn.execute(
                "SELECT tick_ts, status, fallback_triggered, deadline_slip_ms "
                "FROM tick_audit ORDER BY tick_ts DESC LIMIT 1",
            ).fetchone()
        if latest_row is None:
            _check(report, "latest_tick", False, "no tick_audit rows found")
            report.blocking_reasons.append("no_completed_tick")
        else:
            report.last_tick_ts = str(latest_row[0])
            report.last_tick_status = str(latest_row[1])
            report.last_tick_fallback_triggered = _safe_int(latest_row[2])
            report.last_tick_deadline_slip_ms = _safe_int(latest_row[3])
            tick_passed = report.last_tick_status == "COMPLETED"
            _check(
                report, "latest_tick_status",
                tick_passed,
                f"tick {report.last_tick_ts} status={report.last_tick_status}",
            )
            if not tick_passed:
                report.blocking_reasons.append("last_tick_not_completed")
            fallback_passed = (
                report.last_tick_fallback_triggered is not None
                and report.last_tick_fallback_triggered == 0
            )
            _check(
                report, "latest_tick_fallback",
                fallback_passed,
                f"fallback_triggered={report.last_tick_fallback_triggered}",
            )
            if not fallback_passed:
                report.blocking_reasons.append("last_tick_fallback_triggered")
            slip_passed = (
                report.last_tick_deadline_slip_ms is not None
                and report.last_tick_deadline_slip_ms <= deadline_slip_ms_max
            )
            _check(
                report, "latest_tick_deadline_slip",
                slip_passed,
                f"deadline_slip_ms={report.last_tick_deadline_slip_ms} (cap={deadline_slip_ms_max})",
            )
            if not slip_passed:
                report.blocking_reasons.append("last_tick_deadline_slip")

        # --- spend under caps ----------------------------------------------
        spend_total = 0.0
        spend_today = 0.0
        if _table_exists(conn, "spend_log"):
            spend_total = float(
                _scalar(conn, "SELECT coalesce(sum(cost_usd), 0) FROM spend_log") or 0.0
            )
            today_key = datetime.now(tz=UTC).date().isoformat()
            spend_today = float(
                _scalar(
                    conn,
                    "SELECT coalesce(sum(cost_usd), 0) FROM spend_log WHERE day_key=?",
                    (today_key,),
                ) or 0.0
            )
        report.spend_total_usd = spend_total
        report.spend_daily_usd = spend_today
        spend_passed = spend_today <= daily_hard_usd and spend_total <= total_budget_usd
        _check(
            report, "spend_under_caps",
            spend_passed,
            f"daily=${spend_today:.4f} (cap ${daily_hard_usd:.2f}); "
            f"total=${spend_total:.4f} (cap ${total_budget_usd:.2f})",
        )
        if not spend_passed:
            report.blocking_reasons.append("spend_over_cap")

        # --- fill telemetry schema -----------------------------------------
        fills_present = _table_exists(conn, "fills")
        _check(
            report, "fill_telemetry_schema", fills_present,
            "fills table " + ("present" if fills_present else "missing"),
        )
        if fills_present:
            report.fills_count = _safe_int(
                _scalar(conn, "SELECT count(*) FROM fills") or 0,
            ) or 0
            report.fills_filled_count = _safe_int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM fills WHERE fill_status IN ('FILLED','PARTIAL')",
                ) or 0,
            ) or 0
        else:
            report.blocking_reasons.append("fill_telemetry_missing")

        # --- accepted proposals --------------------------------------------
        accepted_count = 0
        if _table_exists(conn, "proposals"):
            accepted_count = _safe_int(
                _scalar(conn, "SELECT count(*) FROM proposals WHERE decision='accept'") or 0,
            ) or 0
        report.accepted_proposals_count = accepted_count
        accepted_passed = accepted_count >= 1
        _check(
            report, "accepted_proposal_exists",
            accepted_passed,
            f"proposals.decision='accept' rows = {accepted_count}",
            severity=("warning" if accepted_count == 0 else "info"),
        )
        if not accepted_passed:
            report.blocking_reasons.append("no_accepted_proposals")

        # --- report modules import + run -----------------------------------
        try:
            from kalibre.reports.fill_report import build_fill_report  # noqa: F401
            from kalibre.reports.shadow_report import build_report  # noqa: F401
        except Exception as exc:
            _check(report, "reports_importable", False, f"import error: {exc}")
            report.blocking_reasons.append("reports_import_failed")
        else:
            _check(report, "reports_importable", True, "fill_report + shadow_report imported")
            try:
                from kalibre.reports.fill_report import build_fill_report as _bfr
                from kalibre.reports.shadow_report import build_report as _br
                _bfr(resolved)
                _br(resolved)
                _check(report, "reports_runnable", True, "fill_report + shadow_report ran against this DB")
            except Exception as exc:
                _check(report, "reports_runnable", False, f"report build error: {exc}")
                report.blocking_reasons.append("reports_build_failed")

        # --- live-canary env gates -----------------------------------------
        env_live = _bool_env(env, "KALIBRE_ENABLE_LIVE_TRADES")
        env_canary = _bool_env(env, "KALIBRE_LIVE_CANARY_MODE")
        try:
            max_intents = int(env.get("KALIBRE_CANARY_MAX_INTENTS") or 0)
        except ValueError:
            max_intents = 0
        try:
            max_size_usd = float(env.get("KALIBRE_CANARY_MAX_SIZE_USD") or 0.0)
        except ValueError:
            max_size_usd = 0.0

        canary_blocking: list[str] = []
        if not env_live:
            canary_blocking.append("KALIBRE_ENABLE_LIVE_TRADES != 1")
        if not env_canary:
            canary_blocking.append("KALIBRE_LIVE_CANARY_MODE != 1")
        if max_intents == 0:
            canary_blocking.append("KALIBRE_CANARY_MAX_INTENTS not set")
        elif max_intents > LIVE_CANARY_MAX_INTENTS_CAP:
            canary_blocking.append(
                f"KALIBRE_CANARY_MAX_INTENTS={max_intents} > cap {LIVE_CANARY_MAX_INTENTS_CAP}"
            )
        if max_size_usd <= 0.0:
            canary_blocking.append("KALIBRE_CANARY_MAX_SIZE_USD not set")
        elif max_size_usd > LIVE_CANARY_MAX_SIZE_USD_CAP:
            canary_blocking.append(
                f"KALIBRE_CANARY_MAX_SIZE_USD={max_size_usd} > cap ${LIVE_CANARY_MAX_SIZE_USD_CAP}"
            )

        env_passed = not canary_blocking
        _check(
            report, "live_canary_env_safe", env_passed,
            "; ".join(canary_blocking) if canary_blocking else "live + canary flags set, caps within limits",
            severity=("warning" if not env_passed else "info"),
        )
        if not env_passed:
            for reason in canary_blocking:
                report.blocking_reasons.append(f"env:{reason}")

        report.suggested_live_canary_command = _build_live_canary_command(slug_for_canary)

        # --- final eligibility ---------------------------------------------
        critical_passed = all(
            c.passed for c in report.checks
            if c.severity == "error" and c.name != "live_canary_env_safe"
        )
        report.live_canary_eligible = bool(
            critical_passed
            and report.accepted_proposals_count >= 1
            and env_passed
        )
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    return report


# --- rendering -------------------------------------------------------------


def render_preflight(report: PreflightReport) -> str:
    lines: list[str] = []
    lines.append(f"# Preflight - {report.db_path}")
    lines.append("")
    lines.append("## Checks")
    for check in report.checks:
        mark = "OK " if check.passed else "FAIL" if check.severity == "error" else "WARN"
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    lines.append("")
    lines.append("## State summary")
    lines.append(f"  last_tick_ts             : {report.last_tick_ts}")
    lines.append(f"  last_tick_status         : {report.last_tick_status}")
    lines.append(f"  last_tick_fallback       : {report.last_tick_fallback_triggered}")
    lines.append(f"  last_tick_deadline_slip  : {report.last_tick_deadline_slip_ms} ms")
    lines.append(f"  spend_daily              : ${report.spend_daily_usd:.4f}")
    lines.append(f"  spend_total              : ${report.spend_total_usd:.4f}")
    lines.append(f"  fills_count              : {report.fills_count} (filled+partial = {report.fills_filled_count})")
    lines.append(f"  accepted_proposals_count : {report.accepted_proposals_count}")
    lines.append("")
    lines.append("## Live canary eligibility")
    lines.append(f"  eligible: {report.live_canary_eligible}")
    if report.blocking_reasons:
        lines.append("  blocking reasons:")
        for reason in report.blocking_reasons:
            lines.append(f"    - {reason}")
    else:
        lines.append("  no blocking reasons")
    lines.append("")
    lines.append("## Env snapshot")
    for key, value in report.env_state.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("## Suggested commands")
    lines.append("  dry-run (rerun a fresh smoke):")
    lines.append(f"    {report.suggested_dry_run_command}")
    if report.live_canary_eligible and report.suggested_live_canary_command:
        lines.append("  live canary (eligible):")
        lines.append(f"    {report.suggested_live_canary_command}")
    elif report.suggested_live_canary_command:
        lines.append("  live canary (BLOCKED -- fix blocking reasons above before running):")
        lines.append(f"    {report.suggested_live_canary_command}")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.preflight")
    parser.add_argument(
        "path",
        help="experiment folder or path to state.sqlite3",
    )
    parser.add_argument("--slug-hint", default=None,
                        help="seed for the suggested run commands")
    parser.add_argument("--json", action="store_true",
                        help="emit a JSON dump instead of human-readable text")
    parser.add_argument("--out", default=None, help="optional output file path")
    args = parser.parse_args(argv)
    report = run_preflight(args.path, slug_hint=args.slug_hint)
    if args.json:
        text = json.dumps(report.to_dict(), indent=2, default=str)
    else:
        text = render_preflight(report)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0 if report.live_canary_eligible else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
