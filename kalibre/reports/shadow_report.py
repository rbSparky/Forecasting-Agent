"""Phase 4A shadow ablation report.

CLI::

    python -m kalibre.reports.shadow_report <experiment_folder_or_state.sqlite3>

Or in code::

    from kalibre.reports.shadow_report import build_report, render_report
    report = build_report(Path("experiments/<slug>__<id>/state.sqlite3"))
    print(render_report(report))

The report is read-only: it opens ``state.sqlite3`` and aggregates the
tables that Phase 4A populates. Designed to compare shadow variants
against the primary path before any layer is promoted to live.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ShadowReport:
    """Aggregated ablation metrics for one experiment DB."""

    db_path: str
    experiment_ids: list[str] = field(default_factory=list)
    candidates_loaded: int = 0
    quote_history_count: int = 0
    quote_features_count: int = 0
    quote_features_warmup_count: int = 0
    quote_features_smart_count: int = 0
    forecasts_count: int = 0
    forecasts_spend_usd: float = 0.0
    proposals_count: int = 0
    proposals_by_decision: dict[str, int] = field(default_factory=dict)
    shadow_proposals_by_variant: dict[str, int] = field(default_factory=dict)
    shadow_proposals_by_variant_decision: dict[str, dict[str, int]] = field(default_factory=dict)
    longshot_by_decision: dict[str, int] = field(default_factory=dict)
    longshot_reject_reasons: dict[str, int] = field(default_factory=dict)
    forecast_selection_by_decision: dict[str, int] = field(default_factory=dict)
    forecast_selection_reject_reasons: dict[str, int] = field(default_factory=dict)
    forecast_selection_selected_ids: list[str] = field(default_factory=list)
    forecast_selection_top_skipped: list[dict[str, Any]] = field(default_factory=list)
    spend_log_count: int = 0
    spend_log_total_usd: float = 0.0
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# --- helpers ---------------------------------------------------------------


def resolve_db_path(path: str | Path) -> Path:
    """Accept either ``state.sqlite3`` directly or an experiment folder."""
    p = Path(path)
    if p.is_dir():
        candidate = p / "state.sqlite3"
        if not candidate.exists():
            raise FileNotFoundError(f"state.sqlite3 not found in {p}")
        return candidate
    return p


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0] or 0)


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = (), default: float = 0.0) -> float:
    row = conn.execute(sql, params).fetchone()
    if row is None or row[0] is None:
        return default
    return float(row[0])


def _group_count(
    conn: sqlite3.Connection,
    table: str,
    *,
    columns: list[str],
    where: str | None = None,
) -> list[tuple]:
    cols_sql = ", ".join(columns)
    sql = f"SELECT {cols_sql}, count(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" GROUP BY {cols_sql}"
    sql += " ORDER BY count(*) DESC"
    return list(conn.execute(sql))


def _examples(
    conn: sqlite3.Connection,
    sql: str,
    *,
    params: tuple = (),
    limit: int = 3,
) -> list[dict[str, Any]]:
    cur = conn.execute(sql + f" LIMIT {int(limit)}", params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# --- core builder ---------------------------------------------------------


def build_report(db_path: str | Path) -> ShadowReport:
    """Aggregate everything Phase 4A needs from ``state.sqlite3``."""
    resolved = resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved))
    try:
        report = ShadowReport(db_path=str(resolved))
        if _has_table(conn, "tick_audit"):
            report.candidates_loaded = _count(
                conn, "SELECT coalesce(sum(candidates_loaded), 0) FROM tick_audit",
            )
            ids = list(conn.execute(
                "SELECT DISTINCT experiment_id FROM tick_audit"
            ))
            report.experiment_ids = sorted({r[0] for r in ids if r[0]})

        if _has_table(conn, "quote_history"):
            report.quote_history_count = _count(
                conn, "SELECT count(*) FROM quote_history",
            )
        if _has_table(conn, "quote_features"):
            report.quote_features_count = _count(
                conn, "SELECT count(*) FROM quote_features",
            )
            report.quote_features_warmup_count = _count(
                conn, "SELECT count(*) FROM quote_features WHERE qh_warmup=1",
            )
            report.quote_features_smart_count = _count(
                conn, "SELECT count(*) FROM quote_features WHERE market_is_smart=1",
            )

        if _has_table(conn, "forecasts"):
            report.forecasts_count = _count(conn, "SELECT count(*) FROM forecasts")
            report.forecasts_spend_usd = _scalar(
                conn, "SELECT coalesce(sum(api_cost_usd), 0) FROM forecasts",
            )

        if _has_table(conn, "proposals"):
            report.proposals_count = _count(conn, "SELECT count(*) FROM proposals")
            report.proposals_by_decision = {
                str(dec): int(c)
                for dec, c in _group_count(conn, "proposals", columns=["decision"])
            }

        if _has_table(conn, "shadow_proposals"):
            variant_counts = _group_count(conn, "shadow_proposals", columns=["variant_name"])
            report.shadow_proposals_by_variant = {
                str(v): int(c) for v, c in variant_counts
            }
            variant_decision = _group_count(
                conn, "shadow_proposals", columns=["variant_name", "decision"],
            )
            by_variant_decision: dict[str, dict[str, int]] = {}
            for variant, decision, count in variant_decision:
                by_variant_decision.setdefault(str(variant), {})[str(decision)] = int(count)
            report.shadow_proposals_by_variant_decision = by_variant_decision
            # Longshot-specific breakdown.
            ls_by_decision = _group_count(
                conn, "shadow_proposals",
                columns=["decision"],
                where="variant_name='longshot'",
            )
            report.longshot_by_decision = {
                str(d): int(c) for d, c in ls_by_decision
            }
            ls_reasons = _group_count(
                conn, "shadow_proposals",
                columns=["reject_reason"],
                where="variant_name='longshot' AND reject_reason IS NOT NULL",
            )
            report.longshot_reject_reasons = {
                str(r): int(c) for r, c in ls_reasons
            }
            # Sample rows per variant for human inspection.
            for variant in sorted(report.shadow_proposals_by_variant):
                report.examples[variant] = _examples(
                    conn,
                    "SELECT variant_name, market_id, decision, reject_reason, "
                    "p_mean, sigma_p, score, audit_json "
                    "FROM shadow_proposals WHERE variant_name=? ORDER BY shadow_id DESC",
                    params=(variant,),
                    limit=2,
                )

            # Phase 4B: dedicated forecast_selection breakdown.
            sel_by_decision = _group_count(
                conn, "shadow_proposals",
                columns=["decision"],
                where="variant_name='forecast_selection'",
            )
            report.forecast_selection_by_decision = {
                str(d): int(c) for d, c in sel_by_decision
            }
            sel_reasons = _group_count(
                conn, "shadow_proposals",
                columns=["reject_reason"],
                where="variant_name='forecast_selection' AND reject_reason IS NOT NULL",
            )
            report.forecast_selection_reject_reasons = {
                str(r): int(c) for r, c in sel_reasons
            }
            sel_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT market_id FROM shadow_proposals "
                    "WHERE variant_name='forecast_selection' AND decision='select' "
                    "ORDER BY score DESC, market_id"
                )
            ]
            report.forecast_selection_selected_ids = sel_ids
            report.forecast_selection_top_skipped = _examples(
                conn,
                "SELECT market_id, reject_reason, score, audit_json "
                "FROM shadow_proposals "
                "WHERE variant_name='forecast_selection' AND decision='skip' "
                "ORDER BY score DESC, market_id",
                limit=5,
            )

        if _has_table(conn, "spend_log"):
            report.spend_log_count = _count(conn, "SELECT count(*) FROM spend_log")
            report.spend_log_total_usd = _scalar(
                conn, "SELECT coalesce(sum(cost_usd), 0) FROM spend_log",
            )

        return report
    finally:
        conn.close()


# --- rendering ------------------------------------------------------------


def render_report(report: ShadowReport) -> str:
    """Pretty-print a :class:`ShadowReport`."""
    lines: list[str] = []
    lines.append(f"# Shadow report - {report.db_path}")
    if report.experiment_ids:
        lines.append(f"experiment_ids: {', '.join(report.experiment_ids)}")
    lines.append("")
    lines.append("## Tick / portfolio")
    lines.append(f"  candidates_loaded (sum across ticks): {report.candidates_loaded}")
    lines.append("")
    lines.append("## Quote-history")
    lines.append(f"  quote_history rows : {report.quote_history_count}")
    lines.append(
        f"  quote_features rows: {report.quote_features_count} "
        f"(warmup={report.quote_features_warmup_count}, smart={report.quote_features_smart_count})"
    )
    lines.append("")
    lines.append("## Forecasts + spend")
    lines.append(
        f"  forecasts: {report.forecasts_count}  "
        f"sum(api_cost_usd)=${report.forecasts_spend_usd:.6f}"
    )
    lines.append(
        f"  spend_log: {report.spend_log_count}  "
        f"sum(cost_usd)=${report.spend_log_total_usd:.6f}"
    )
    lines.append("")
    lines.append("## Proposals")
    lines.append(f"  total: {report.proposals_count}")
    for dec, c in sorted(report.proposals_by_decision.items()):
        lines.append(f"    decision={dec}: {c}")
    lines.append("")
    lines.append("## Shadow proposals by variant")
    if not report.shadow_proposals_by_variant:
        lines.append("  (none)")
    for variant, c in sorted(report.shadow_proposals_by_variant.items()):
        lines.append(f"  {variant}: {c}")
        for decision, dcount in sorted(report.shadow_proposals_by_variant_decision.get(variant, {}).items()):
            lines.append(f"    decision={decision}: {dcount}")
    if report.longshot_by_decision:
        lines.append("")
        lines.append("## Longshot variant breakdown")
        for dec, c in sorted(report.longshot_by_decision.items()):
            lines.append(f"  decision={dec}: {c}")
        for reason, c in sorted(report.longshot_reject_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    reject_reason={reason}: {c}")
    if report.forecast_selection_by_decision:
        lines.append("")
        lines.append("## Forecast selection breakdown")
        for dec, c in sorted(report.forecast_selection_by_decision.items()):
            lines.append(f"  decision={dec}: {c}")
        for reason, c in sorted(report.forecast_selection_reject_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    reject_reason={reason}: {c}")
        if report.forecast_selection_selected_ids:
            preview = ", ".join(report.forecast_selection_selected_ids[:12])
            lines.append(f"  selected_market_ids: {preview}")
        if report.forecast_selection_top_skipped:
            lines.append("  top skipped (highest score among skips):")
            for row in report.forecast_selection_top_skipped:
                meta = ", ".join(
                    f"{k}={row.get(k)}" for k in ("market_id", "reject_reason", "score")
                    if row.get(k) is not None
                )
                lines.append(f"    {meta}")
    if report.examples:
        lines.append("")
        lines.append("## Example shadow rows (most recent first)")
        for variant in sorted(report.examples):
            lines.append(f"  -- {variant} --")
            for row in report.examples[variant]:
                meta = ", ".join(
                    f"{k}={row.get(k)}" for k in ("market_id", "decision", "reject_reason", "p_mean", "sigma_p", "score")
                    if row.get(k) is not None
                )
                lines.append(f"    {meta}")
                aj = row.get("audit_json")
                if isinstance(aj, str):
                    try:
                        compact = json.dumps(json.loads(aj), sort_keys=True)[:240]
                        lines.append(f"      audit_json: {compact}")
                    except json.JSONDecodeError:
                        lines.append(f"      audit_json: {aj[:240]}")
    return "\n".join(lines)


def report_to_dict(report: ShadowReport) -> dict[str, Any]:
    return asdict(report)


# --- CLI ------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.reports.shadow_report")
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
    report = build_report(args.path)
    if args.json:
        text = json.dumps(report_to_dict(report), indent=2, default=str)
    else:
        text = render_report(report)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
