"""Phase 6B evidence report.

Pure read-only summary of the new ``web_search_queries`` /
``web_search_results`` / ``evidence_bundles`` tables, plus
edge-source-aware aggregates over the ``forecasts`` table.

CLI::

    python -m kalibre.reports.evidence_report <folder_or_state.sqlite3>
    python -m kalibre.reports.evidence_report <path> --json --out evidence.json

Tolerates Phase-5B-era DBs that have no Phase 6B tables (returns
all-zero counters).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvidenceReport:
    db_path: str
    experiment_ids: list[str] = field(default_factory=list)
    web_search_queries_count: int = 0
    web_search_results_count: int = 0
    evidence_bundles_count: int = 0
    total_web_search_requests: int = 0
    total_search_spend_usd: float = 0.0
    forecasts_by_edge_source: dict[str, int] = field(default_factory=dict)
    domains_by_count: list[tuple[str, int]] = field(default_factory=list)
    domains_by_tier: dict[str, int] = field(default_factory=dict)
    evidence_quality_buckets: dict[str, int] = field(default_factory=dict)
    evidence_quality_avg: float | None = None
    stale_evidence_count: int = 0
    forecasts_with_evidence: int = 0
    forecasts_without_evidence: int = 0
    # Phase 6C polish: split forecasts-with-evidence into fresh
    # (cache_hit=0) vs cached replays (cache_hit=1) so reports
    # distinguish "we paid for a new search" from "we reused prior
    # evidence". `total_web_search_requests` already sums only the
    # evidence_bundles rows (which the runner skips on cache hits), so
    # it represents fresh search activity.
    forecasts_with_evidence_fresh: int = 0
    forecasts_with_evidence_cached: int = 0
    cited_sample: list[dict[str, Any]] = field(default_factory=list)


# --- helpers --------------------------------------------------------------


def resolve_db_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        candidate = p / "state.sqlite3"
        if not candidate.exists():
            raise FileNotFoundError(f"state.sqlite3 not found in {p}")
        return candidate
    return p


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = (), default: Any = 0) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None or row[0] is None:
        return default
    return row[0]


def build_evidence_report(db_path: str | Path) -> EvidenceReport:
    resolved = resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved))
    try:
        report = EvidenceReport(db_path=str(resolved))
        if _has_table(conn, "tick_audit"):
            report.experiment_ids = sorted({
                r[0] for r in conn.execute("SELECT DISTINCT experiment_id FROM tick_audit")
                if r[0]
            })

        # Phase 6B audit tables. Pre-6B DBs simply have zero counts.
        if _has_table(conn, "web_search_queries"):
            report.web_search_queries_count = int(
                _scalar(conn, "SELECT count(*) FROM web_search_queries", default=0)
            )
        if _has_table(conn, "web_search_results"):
            report.web_search_results_count = int(
                _scalar(conn, "SELECT count(*) FROM web_search_results", default=0)
            )
            rows = list(conn.execute(
                "SELECT domain, count(*) FROM web_search_results "
                "WHERE domain IS NOT NULL AND domain<>'' "
                "GROUP BY domain ORDER BY count(*) DESC LIMIT 12"
            ))
            report.domains_by_count = [(str(d), int(c)) for d, c in rows]
            tier_rows = list(conn.execute(
                "SELECT source_tier, count(*) FROM web_search_results "
                "WHERE source_tier IS NOT NULL "
                "GROUP BY source_tier ORDER BY source_tier"
            ))
            report.domains_by_tier = {str(t): int(c) for t, c in tier_rows}
            # Sample (small) for human inspection.
            sample = list(conn.execute(
                "SELECT market_id, url, domain, source_tier FROM web_search_results "
                "ORDER BY result_id DESC LIMIT 8"
            ))
            report.cited_sample = [
                {
                    "market_id": str(r[0]), "url": str(r[1]),
                    "domain": str(r[2]) if r[2] else "",
                    "source_tier": str(r[3]) if r[3] else "",
                }
                for r in sample
            ]
        if _has_table(conn, "evidence_bundles"):
            report.evidence_bundles_count = int(
                _scalar(conn, "SELECT count(*) FROM evidence_bundles", default=0)
            )
            report.total_web_search_requests = int(
                _scalar(
                    conn,
                    "SELECT coalesce(sum(web_search_requests), 0) FROM evidence_bundles",
                    default=0,
                )
            )
            avg = _scalar(
                conn,
                "SELECT avg(evidence_quality) FROM evidence_bundles "
                "WHERE evidence_quality IS NOT NULL",
                default=None,
            )
            if avg is not None:
                report.evidence_quality_avg = round(float(avg), 4)
            report.stale_evidence_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM evidence_bundles WHERE stale_evidence=1",
                    default=0,
                )
            )
            # Quality buckets: low (<0.3) / med (0.3-0.7) / high (>0.7).
            buckets_row = conn.execute(
                "SELECT "
                "  sum(case when evidence_quality<0.3 then 1 else 0 end), "
                "  sum(case when evidence_quality>=0.3 and evidence_quality<=0.7 then 1 else 0 end), "
                "  sum(case when evidence_quality>0.7 then 1 else 0 end) "
                "FROM evidence_bundles WHERE evidence_quality IS NOT NULL"
            ).fetchone() or (0, 0, 0)
            report.evidence_quality_buckets = {
                "low_lt_0.3": int(buckets_row[0] or 0),
                "med_0.3_0.7": int(buckets_row[1] or 0),
                "high_gt_0.7": int(buckets_row[2] or 0),
            }

        # Search spend (separate purpose='web_search').
        if _has_table(conn, "spend_log"):
            report.total_search_spend_usd = float(
                _scalar(
                    conn,
                    "SELECT coalesce(sum(cost_usd), 0) FROM spend_log WHERE purpose='web_search'",
                    default=0.0,
                )
            )

        # Edge-source split (counts per forecasts row's edge_source).
        if _has_table(conn, "forecasts"):
            rows = list(conn.execute(
                "SELECT edge_source, count(*) FROM forecasts "
                "WHERE edge_source IS NOT NULL "
                "GROUP BY edge_source ORDER BY count(*) DESC"
            ))
            report.forecasts_by_edge_source = {str(e): int(c) for e, c in rows}
            # Phase 6C repair: count evidence-bearing forecasts by
            # actually parsing audit_json and checking ``cited_urls``
            # length AND ``web_search_requests`` (>0). The previous
            # substring heuristic ``%"web_search_requests": 0%`` was
            # wrong for engines that return citations but no
            # server_tool_use field (web_search_requests can be 0 with
            # citations present -- see provider's inferred-count path).
            forecast_rows = list(conn.execute(
                "SELECT audit_json, cache_hit FROM forecasts "
                "WHERE audit_json IS NOT NULL"
            ))
            with_evidence = 0
            without_evidence = 0
            with_evidence_fresh = 0
            with_evidence_cached = 0
            for aj_raw, cache_hit_col in forecast_rows:
                try:
                    aj = json.loads(aj_raw or "{}")
                except (TypeError, ValueError):
                    aj = {}
                requests_n = int(aj.get("web_search_requests") or 0)
                cited = aj.get("cited_urls") or []
                is_cached = bool(cache_hit_col) or bool(aj.get("cache_hit"))
                if requests_n > 0 or (isinstance(cited, list) and len(cited) > 0):
                    with_evidence += 1
                    if is_cached:
                        with_evidence_cached += 1
                    else:
                        with_evidence_fresh += 1
                else:
                    without_evidence += 1
            report.forecasts_with_evidence = with_evidence
            report.forecasts_without_evidence = without_evidence
            report.forecasts_with_evidence_fresh = with_evidence_fresh
            report.forecasts_with_evidence_cached = with_evidence_cached

        return report
    finally:
        conn.close()


# --- rendering ------------------------------------------------------------


def render_evidence_report(report: EvidenceReport) -> str:
    lines: list[str] = []
    lines.append(f"# Evidence report - {report.db_path}")
    if report.experiment_ids:
        lines.append(f"experiment_ids: {', '.join(report.experiment_ids)}")
    lines.append("")
    lines.append("## 1. Search activity")
    lines.append(f"  web_search_queries rows : {report.web_search_queries_count}")
    lines.append(f"  web_search_results rows : {report.web_search_results_count}")
    lines.append(f"  evidence_bundles rows   : {report.evidence_bundles_count}")
    lines.append(f"  total search requests    : {report.total_web_search_requests}")
    lines.append(f"  total search spend USD   : ${report.total_search_spend_usd:.4f}")
    lines.append("")
    lines.append("## 2. Source tier mix")
    if report.domains_by_tier:
        for tier in ("A", "B", "C", "D"):
            c = report.domains_by_tier.get(tier, 0)
            lines.append(f"  tier {tier}: {c}")
    else:
        lines.append("  (no cited URLs)")
    lines.append("")
    lines.append("## 3. Top domains")
    if report.domains_by_count:
        for dom, c in report.domains_by_count:
            lines.append(f"  {dom}: {c}")
    else:
        lines.append("  (no cited URLs)")
    lines.append("")
    lines.append("## 4. Evidence quality")
    lines.append(f"  avg evidence_quality : {report.evidence_quality_avg}")
    lines.append(f"  stale_evidence rows  : {report.stale_evidence_count}")
    if report.evidence_quality_buckets:
        for k in ("low_lt_0.3", "med_0.3_0.7", "high_gt_0.7"):
            lines.append(f"  {k}: {report.evidence_quality_buckets.get(k, 0)}")
    lines.append("")
    lines.append("## 5. Forecast edge sources")
    if report.forecasts_by_edge_source:
        for e, c in sorted(
            report.forecasts_by_edge_source.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {e}: {c}")
    else:
        lines.append("  (no forecasts)")
    lines.append(
        f"  forecasts_with_evidence    : {report.forecasts_with_evidence} "
        f"(fresh={report.forecasts_with_evidence_fresh}, "
        f"cached={report.forecasts_with_evidence_cached})"
    )
    lines.append(
        f"  forecasts_without_evidence : {report.forecasts_without_evidence}"
    )
    if report.cited_sample:
        lines.append("")
        lines.append("## 6. Sample cited URLs")
        for row in report.cited_sample[:8]:
            lines.append(
                f"  [{row.get('source_tier','?')}] {row.get('market_id','')}  "
                f"{row.get('url','')}"
            )
    return "\n".join(lines)


def report_to_dict(report: EvidenceReport) -> dict[str, Any]:
    return asdict(report)


# --- CLI ------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.reports.evidence_report")
    parser.add_argument("path", help="experiment folder or path to state.sqlite3")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--out", default=None, help="output file (default stdout)")
    args = parser.parse_args(argv)
    report = build_evidence_report(args.path)
    if args.json:
        text = json.dumps(report_to_dict(report), indent=2, default=str)
    else:
        text = render_evidence_report(report)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
