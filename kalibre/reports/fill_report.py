"""Phase 5A fill / rejection telemetry report.

CLI::

    python -m kalibre.reports.fill_report <experiment_folder_or_state.sqlite3>

Or in code::

    from kalibre.reports.fill_report import build_fill_report, render_fill_report
    report = build_fill_report(Path("experiments/<slug>__<id>/state.sqlite3"))
    print(render_fill_report(report))

Aggregates the ``fills`` table populated by the Phase 5A audit hooks.
Read-only; safe to run alongside the live loop.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --- dataclass ------------------------------------------------------------


@dataclass
class FillReport:
    db_path: str
    submitted: int = 0
    filled: int = 0
    partial: int = 0
    rejected: int = 0
    unknown: int = 0
    expected_fill_prob_avg: float | None = None
    realized_fill_rate: float | None = None
    slippage_bps_avg: float | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    by_edge_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# --- helpers --------------------------------------------------------------


def resolve_db_path(path: str | Path) -> Path:
    """Accept either ``state.sqlite3`` directly or an experiment folder."""
    p = Path(path)
    if p.is_dir():
        candidate = p / "state.sqlite3"
        if not candidate.exists():
            raise FileNotFoundError(f"state.sqlite3 not found in {p}")
        return candidate
    return p


def _has_fills_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fills'"
    ).fetchone()
    return row is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return row[0]


# --- builder --------------------------------------------------------------


def build_fill_report(db_path: str | Path) -> FillReport:
    resolved = resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved))
    try:
        report = FillReport(db_path=str(resolved))
        if not _has_fills_table(conn):
            return report
        report.submitted = int(_scalar(conn, "SELECT count(*) FROM fills") or 0)
        if report.submitted == 0:
            return report

        # Status counts.
        for status, count in conn.execute(
            "SELECT fill_status, count(*) FROM fills GROUP BY fill_status"
        ):
            status_norm = (status or "UNKNOWN").upper()
            if status_norm == "FILLED":
                report.filled = int(count)
            elif status_norm == "PARTIAL":
                report.partial = int(count)
            elif status_norm == "REJECTED":
                report.rejected = int(count)
            else:
                report.unknown = int(count)

        # Averages.
        exp_avg = _scalar(
            conn, "SELECT avg(expected_fill_prob) FROM fills WHERE expected_fill_prob IS NOT NULL",
        )
        report.expected_fill_prob_avg = float(exp_avg) if exp_avg is not None else None

        slip_avg = _scalar(
            conn, "SELECT avg(slippage_bps) FROM fills WHERE slippage_bps IS NOT NULL",
        )
        report.slippage_bps_avg = float(slip_avg) if slip_avg is not None else None

        filled_or_partial = report.filled + report.partial
        denom = report.submitted - report.unknown
        if denom > 0:
            report.realized_fill_rate = filled_or_partial / denom

        # Rejection reasons.
        for reason, count in conn.execute(
            "SELECT rejection_reason, count(*) FROM fills "
            "WHERE rejection_reason IS NOT NULL AND rejection_reason != '' "
            "GROUP BY rejection_reason ORDER BY count(*) DESC"
        ):
            report.rejection_reasons[str(reason)] = int(count)

        # Per edge_source breakdown. The status / count split is read
        # from the (edge_source, fill_status) group, but the averages
        # come from a separate aggregate keyed on edge_source ONLY so
        # they don't get overwritten by the last status group.
        for row in conn.execute(
            "SELECT coalesce(edge_source, 'unknown'), fill_status, count(*) "
            "FROM fills GROUP BY edge_source, fill_status"
        ):
            edge_source = str(row[0])
            status = str(row[1] or "UNKNOWN").upper()
            count = int(row[2])
            bucket = report.by_edge_source.setdefault(edge_source, {
                "submitted": 0,
                "filled": 0,
                "partial": 0,
                "rejected": 0,
                "unknown": 0,
                "slippage_bps_avg": None,
                "expected_fill_prob_avg": None,
            })
            bucket["submitted"] += count
            if status == "FILLED":
                bucket["filled"] = count
            elif status == "PARTIAL":
                bucket["partial"] = count
            elif status == "REJECTED":
                bucket["rejected"] = count
            else:
                bucket["unknown"] = count
        for row in conn.execute(
            "SELECT coalesce(edge_source, 'unknown'), "
            "avg(slippage_bps), avg(expected_fill_prob) "
            "FROM fills GROUP BY edge_source"
        ):
            edge_source = str(row[0])
            slip = float(row[1]) if row[1] is not None else None
            efp = float(row[2]) if row[2] is not None else None
            bucket = report.by_edge_source.setdefault(edge_source, {
                "submitted": 0,
                "filled": 0,
                "partial": 0,
                "rejected": 0,
                "unknown": 0,
                "slippage_bps_avg": None,
                "expected_fill_prob_avg": None,
            })
            bucket["slippage_bps_avg"] = slip
            bucket["expected_fill_prob_avg"] = efp

        # Example rows per status.
        for status in ("FILLED", "PARTIAL", "REJECTED", "UNKNOWN"):
            cur = conn.execute(
                "SELECT intent_id, market_id, side, submitted_shares, "
                "submitted_price_implied, filled_shares, filled_price, "
                "slippage_bps, rejection_reason, audit_json "
                "FROM fills WHERE fill_status=? ORDER BY fill_id DESC LIMIT 2",
                (status,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            if rows:
                report.examples[status] = rows
        return report
    finally:
        conn.close()


# --- rendering ------------------------------------------------------------


def render_fill_report(report: FillReport) -> str:
    lines: list[str] = []
    lines.append(f"# Fill report - {report.db_path}")
    lines.append("")
    lines.append("## Submission counts")
    lines.append(f"  submitted : {report.submitted}")
    if report.submitted == 0:
        lines.append("  (no fill rows; nothing else to summarize)")
        return "\n".join(lines)
    lines.append(f"  filled    : {report.filled}")
    lines.append(f"  partial   : {report.partial}")
    lines.append(f"  rejected  : {report.rejected}")
    lines.append(f"  unknown   : {report.unknown}")
    lines.append("")
    lines.append("## Averages")
    if report.expected_fill_prob_avg is not None:
        lines.append(f"  expected_fill_prob (avg) : {report.expected_fill_prob_avg:.4f}")
    else:
        lines.append("  expected_fill_prob (avg) : n/a")
    if report.realized_fill_rate is not None:
        lines.append(f"  realized_fill_rate       : {report.realized_fill_rate:.4f}")
    else:
        lines.append("  realized_fill_rate       : n/a")
    if report.slippage_bps_avg is not None:
        lines.append(f"  slippage_bps (avg)       : {report.slippage_bps_avg:.4f}")
    else:
        lines.append("  slippage_bps (avg)       : n/a")
    if report.rejection_reasons:
        lines.append("")
        lines.append("## Rejection reasons")
        for reason, count in sorted(report.rejection_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")
    if report.by_edge_source:
        lines.append("")
        lines.append("## By edge_source")
        for source in sorted(report.by_edge_source):
            bucket = report.by_edge_source[source]
            lines.append(f"  {source}")
            for key in ("submitted", "filled", "partial", "rejected", "unknown"):
                lines.append(f"    {key}: {bucket.get(key, 0)}")
            if bucket.get("slippage_bps_avg") is not None:
                lines.append(f"    slippage_bps (avg): {bucket['slippage_bps_avg']:.4f}")
            if bucket.get("expected_fill_prob_avg") is not None:
                lines.append(f"    expected_fill_prob (avg): {bucket['expected_fill_prob_avg']:.4f}")
    if report.examples:
        lines.append("")
        lines.append("## Example rows (most recent first)")
        for status in sorted(report.examples):
            lines.append(f"  -- {status} --")
            for row in report.examples[status]:
                meta = ", ".join(
                    f"{k}={row.get(k)}"
                    for k in (
                        "intent_id", "market_id", "side",
                        "submitted_shares", "submitted_price_implied",
                        "filled_shares", "filled_price", "slippage_bps",
                        "rejection_reason",
                    )
                    if row.get(k) is not None
                )
                lines.append(f"    {meta}")
    return "\n".join(lines)


def report_to_dict(report: FillReport) -> dict[str, Any]:
    return asdict(report)


# --- CLI ------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.reports.fill_report")
    parser.add_argument(
        "path",
        help="experiment folder or path to state.sqlite3",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit a JSON dump instead of human-readable text",
    )
    parser.add_argument(
        "--out", default=None,
        help="optional output file (default: stdout)",
    )
    args = parser.parse_args(argv)
    report = build_fill_report(args.path)
    if args.json:
        text = json.dumps(report_to_dict(report), indent=2, default=str)
    else:
        text = render_fill_report(report)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
