"""Phase 6 trade-production funnel report.

Shows where opportunities die per tick. Read-only -- queries the existing
SQLite tables and produces a per-stage view from candidates loaded through
fills. Handy when the bottleneck shifts (e.g. exploration mode moves the
loss from "0 forecasts" to "Kelly/score rejects everything"). Tolerates
old (pre-Phase 6) DBs that lack newer rows.

CLI::

    python -m kalibre.reports.funnel_report <experiment_folder_or_state.sqlite3>
    python -m kalibre.reports.funnel_report <path> --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FunnelReport:
    db_path: str
    experiment_ids: list[str] = field(default_factory=list)
    candidates_loaded: int = 0
    universe_accepted: int = 0
    universe_rejected: int = 0
    universe_reject_reasons: dict[str, int] = field(default_factory=dict)
    selector_selected: int = 0
    selector_skipped: int = 0
    selector_skip_reasons: dict[str, int] = field(default_factory=dict)
    exploration_promotions: int = 0
    exploration_promotion_ids: list[str] = field(default_factory=list)
    horizon_extended_count: int = 0
    forecasts_by_model_tier: dict[str, int] = field(default_factory=dict)
    forecasts_spend_by_model_tier: dict[str, float] = field(default_factory=dict)
    forecasts_total: int = 0
    forecasts_spend_total_usd: float = 0.0
    proposals_by_edge_source: dict[str, int] = field(default_factory=dict)
    proposals_by_decision: dict[str, int] = field(default_factory=dict)
    proposals_reject_reasons: dict[str, int] = field(default_factory=dict)
    accepted_proposals_count: int = 0
    intents_submitted: int = 0
    fills_by_status: dict[str, int] = field(default_factory=dict)
    # Phase 6R additions.
    fill_prob_below_skip_count: int = 0
    canary_fill_rescue_shadow_count: int = 0
    # Phase 6C polish: Kelly rescue is a distinct shadow stream
    # (variant_name='canary_kelly_rescue_shadow') with its own audit
    # extras. Render separately so the operator can tell which path
    # produced a candidate.
    canary_kelly_rescue_shadow_count: int = 0
    top_fill_rescue_rows: list[dict[str, Any]] = field(default_factory=list)
    top_kelly_rescue_rows: list[dict[str, Any]] = field(default_factory=list)
    quote_age_buckets: dict[str, int] = field(default_factory=dict)
    top_blocked_proposals: list[dict[str, Any]] = field(default_factory=list)
    # Phase 6B web-search stage.
    web_search_requests_total: int = 0
    cited_urls_total: int = 0
    web_sonnet_forecast_count: int = 0
    web_opus_forecast_count: int = 0
    evidence_quality_avg: float | None = None
    # Phase 6D budget-aware routing surface. Derived from forecasts +
    # tick_audit + spend_log. ``budget_profile`` is read from the
    # most-recent tick_audit row's environment snapshot when present;
    # the other counters are SQL aggregates over the new edge_source
    # labels.
    budget_profile: str | None = None
    per_tick_paid_forecast_cap_usd: float | None = None
    forecast_blocked_tick_budget_count: int = 0
    opus_skipped_budget_count: int = 0
    avg_tick_paid_spend_usd: float = 0.0
    projected_14_day_spend_usd: float = 0.0
    ticks_observed: int = 0


# --- helpers ---------------------------------------------------------------


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


# --- builder ---------------------------------------------------------------


def build_funnel_report(db_path: str | Path) -> FunnelReport:
    resolved = resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved))
    try:
        report = FunnelReport(db_path=str(resolved))

        if _has_table(conn, "tick_audit"):
            report.candidates_loaded = int(
                _scalar(conn, "SELECT coalesce(sum(candidates_loaded), 0) FROM tick_audit", default=0)
            )
            report.experiment_ids = sorted({
                r[0] for r in conn.execute("SELECT DISTINCT experiment_id FROM tick_audit")
                if r[0]
            })
            report.intents_submitted = int(
                _scalar(conn, "SELECT coalesce(sum(intents_submitted), 0) FROM tick_audit", default=0)
            )

        # Universe accept/reject derived from proposals audit_rows
        # (strategy.py writes one ``proposals`` row per universe-accepted
        # market and one ``universe_filter`` reject row per rejected
        # market, so we can split on edge_source there).
        if _has_table(conn, "proposals"):
            uni_rej = list(conn.execute(
                "SELECT reject_reason, count(*) FROM proposals "
                "WHERE edge_source='universe_filter' AND reject_reason IS NOT NULL "
                "GROUP BY reject_reason ORDER BY count(*) DESC"
            ))
            report.universe_reject_reasons = {str(r): int(c) for r, c in uni_rej}
            report.universe_rejected = sum(report.universe_reject_reasons.values())
            report.universe_accepted = max(
                0,
                int(
                    _scalar(conn, "SELECT count(*) FROM proposals WHERE edge_source!='universe_filter'", default=0)
                )
            )
            # Proposals broken down by edge_source + decision.
            es = list(conn.execute(
                "SELECT edge_source, count(*) FROM proposals "
                "WHERE edge_source!='universe_filter' "
                "GROUP BY edge_source ORDER BY count(*) DESC"
            ))
            report.proposals_by_edge_source = {str(s): int(c) for s, c in es}
            decisions = list(conn.execute(
                "SELECT decision, count(*) FROM proposals "
                "WHERE edge_source!='universe_filter' "
                "GROUP BY decision ORDER BY count(*) DESC"
            ))
            report.proposals_by_decision = {str(d): int(c) for d, c in decisions}
            report.accepted_proposals_count = int(
                _scalar(conn, "SELECT count(*) FROM proposals WHERE decision='accept'", default=0)
            )
            rj = list(conn.execute(
                "SELECT reject_reason, count(*) FROM proposals "
                "WHERE decision='reject' AND edge_source!='universe_filter' AND reject_reason IS NOT NULL "
                "GROUP BY reject_reason ORDER BY count(*) DESC"
            ))
            report.proposals_reject_reasons = {str(r): int(c) for r, c in rj}

        if _has_table(conn, "shadow_proposals"):
            # Selector rows live under variant_name='forecast_selection'.
            sel = list(conn.execute(
                "SELECT decision, count(*) FROM shadow_proposals "
                "WHERE variant_name='forecast_selection' "
                "GROUP BY decision ORDER BY count(*) DESC"
            ))
            for decision, count in sel:
                if str(decision) == "select":
                    report.selector_selected = int(count)
                else:
                    report.selector_skipped = int(count)
            skip_reasons = list(conn.execute(
                "SELECT reject_reason, count(*) FROM shadow_proposals "
                "WHERE variant_name='forecast_selection' AND decision='skip' "
                "AND reject_reason IS NOT NULL "
                "GROUP BY reject_reason ORDER BY count(*) DESC"
            ))
            report.selector_skip_reasons = {str(r): int(c) for r, c in skip_reasons}
            # Phase 6: exploration promotions surfaced via audit_json.
            promo_rows = list(conn.execute(
                "SELECT market_id, audit_json FROM shadow_proposals "
                "WHERE variant_name='forecast_selection' AND decision='select' "
                "ORDER BY score DESC, market_id"
            ))
            for market_id, audit_json in promo_rows:
                if not audit_json:
                    continue
                try:
                    payload = json.loads(audit_json)
                except (TypeError, ValueError):
                    continue
                if payload.get("selection_reason") == "selected_efficient_exploration":
                    report.exploration_promotion_ids.append(str(market_id))
            report.exploration_promotions = len(report.exploration_promotion_ids)
            # Horizon-extended diagnostic stream count.
            report.horizon_extended_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM shadow_proposals "
                    "WHERE variant_name='universe_horizon_extended'",
                    default=0,
                )
            )

        if _has_table(conn, "forecasts"):
            rows = list(conn.execute(
                "SELECT model_tier, count(*), coalesce(sum(api_cost_usd),0) "
                "FROM forecasts WHERE model_tier IS NOT NULL "
                "GROUP BY model_tier ORDER BY count(*) DESC"
            ))
            for tier, count, cost in rows:
                report.forecasts_by_model_tier[str(tier)] = int(count)
                report.forecasts_spend_by_model_tier[str(tier)] = float(cost or 0.0)
            report.forecasts_total = int(
                _scalar(conn, "SELECT count(*) FROM forecasts", default=0)
            )
            report.forecasts_spend_total_usd = float(
                _scalar(conn, "SELECT coalesce(sum(api_cost_usd), 0) FROM forecasts", default=0.0)
            )

        if _has_table(conn, "fills"):
            rows = list(conn.execute(
                "SELECT fill_status, count(*) FROM fills "
                "WHERE fill_status IS NOT NULL "
                "GROUP BY fill_status ORDER BY count(*) DESC"
            ))
            report.fills_by_status = {str(s): int(c) for s, c in rows}

        # --- Phase 6R: fill-prob blocks, rescue shadows, quote_age dist,
        #     and the top blocked proposals (for canary-rescue triage).
        if _has_table(conn, "proposals"):
            report.fill_prob_below_skip_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM proposals "
                    "WHERE decision='reject' AND reject_reason='fill_prob_below_skip'",
                    default=0,
                )
            )
            # Quote-age distribution + top blocked proposals come from
            # the audit_json column. Parse just the rejected fill_prob
            # rows to keep this cheap.
            blocked_rows = list(conn.execute(
                "SELECT market_id, score, size_usd, shares, fill_prob, audit_json "
                "FROM proposals "
                "WHERE decision='reject' AND reject_reason='fill_prob_below_skip' "
                "ORDER BY score DESC, market_id LIMIT 20"
            ))
            buckets: dict[str, int] = {
                "<=60s": 0, "60-180s": 0, "180-300s": 0, ">300s": 0, "missing": 0,
            }
            top_blocked: list[dict[str, Any]] = []
            for market_id, score, size_usd, shares, fp, audit_json in blocked_rows:
                payload: dict[str, Any] = {}
                if audit_json:
                    try:
                        payload = json.loads(audit_json)
                    except (TypeError, ValueError):
                        payload = {}
                qa = payload.get("quote_age_sec")
                if qa is None:
                    buckets["missing"] += 1
                else:
                    try:
                        qa_f = float(qa)
                        if qa_f <= 60:
                            buckets["<=60s"] += 1
                        elif qa_f <= 180:
                            buckets["60-180s"] += 1
                        elif qa_f <= 300:
                            buckets["180-300s"] += 1
                        else:
                            buckets[">300s"] += 1
                    except (TypeError, ValueError):
                        buckets["missing"] += 1
                top_blocked.append({
                    "market_id": str(market_id),
                    "score": float(score or 0.0),
                    "size_usd": float(size_usd or 0.0),
                    "shares": float(shares or 0.0),
                    "fill_prob": float(fp or 0.0),
                    "original_fill_prob": payload.get("original_fill_prob", float(fp or 0.0)),
                    "canary_fill_prob_25": payload.get("canary_fill_prob_25"),
                    "canary_fill_prob_50": payload.get("canary_fill_prob_50"),
                    "rescue_expected_fill_prob": payload.get("rescue_expected_fill_prob"),
                    "rescue_applied": bool(payload.get("canary_fill_rescue_applied", False)),
                    "quote_age_sec": payload.get("quote_age_sec"),
                    "edge_source": payload.get("edge_source"),
                })
            report.quote_age_buckets = buckets
            report.top_blocked_proposals = top_blocked

        if _has_table(conn, "shadow_proposals"):
            report.canary_fill_rescue_shadow_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM shadow_proposals "
                    "WHERE variant_name='canary_fill_rescue_shadow'",
                    default=0,
                )
            )
            report.canary_kelly_rescue_shadow_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM shadow_proposals "
                    "WHERE variant_name='canary_kelly_rescue_shadow'",
                    default=0,
                )
            )
            for variant, target in (
                ("canary_fill_rescue_shadow", report.top_fill_rescue_rows),
                ("canary_kelly_rescue_shadow", report.top_kelly_rescue_rows),
            ):
                rows = list(conn.execute(
                    "SELECT market_id, side, edge_source, score, audit_json "
                    "FROM shadow_proposals WHERE variant_name=? "
                    "ORDER BY score DESC, market_id LIMIT 5",
                    (variant,),
                ))
                for market_id, side, edge_source, score, audit_json in rows:
                    payload: dict[str, Any] = {}
                    if audit_json:
                        try:
                            payload = json.loads(audit_json)
                        except (TypeError, ValueError):
                            payload = {}
                    target.append({
                        "market_id": str(market_id),
                        "side": str(side) if side is not None else "",
                        "edge_source": str(edge_source or payload.get("edge_source") or ""),
                        "score": float(score or 0.0),
                        "rescue_edge_pp": payload.get("rescue_edge_pp"),
                        "canary_fill_prob_25": payload.get("canary_fill_prob_25"),
                        "canary_fill_prob_50": payload.get("canary_fill_prob_50"),
                        "rescue_expected_fill_prob": payload.get("rescue_expected_fill_prob"),
                        "original_fill_prob": payload.get("original_fill_prob"),
                        "quote_age_sec": payload.get("quote_age_sec"),
                    })

        # Phase 6B: web-search stage counters.
        if _has_table(conn, "evidence_bundles"):
            report.web_search_requests_total = int(
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
        if _has_table(conn, "web_search_results"):
            report.cited_urls_total = int(
                _scalar(conn, "SELECT count(*) FROM web_search_results", default=0)
            )
        if _has_table(conn, "forecasts"):
            report.web_sonnet_forecast_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM forecasts WHERE edge_source='web_sonnet_forecast'",
                    default=0,
                )
            )
            report.web_opus_forecast_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM forecasts WHERE edge_source='web_opus_forecast'",
                    default=0,
                )
            )
            # Phase 6D budget-aware counters. Edge-source labels were
            # added in 6D so older DBs return 0 (safe).
            report.forecast_blocked_tick_budget_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM forecasts "
                    "WHERE edge_source='forecast_blocked_tick_budget'",
                    default=0,
                )
            )
            report.opus_skipped_budget_count = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM forecasts "
                    "WHERE edge_source='opus_skipped_budget_profile'",
                    default=0,
                )
            )
            # Pull budget_profile + cap from the most recent forecast
            # row's audit_json so reports show "what the runner thought
            # the budget was on the last tick". Both are optional --
            # pre-6D rows simply don't have them.
            latest_aj_row = conn.execute(
                "SELECT audit_json FROM forecasts "
                "WHERE audit_json IS NOT NULL "
                "ORDER BY tick_ts DESC LIMIT 1"
            ).fetchone()
            if latest_aj_row and latest_aj_row[0]:
                try:
                    latest_aj = json.loads(latest_aj_row[0])
                except (TypeError, ValueError):
                    latest_aj = {}
                bp = latest_aj.get("budget_profile") if isinstance(latest_aj, dict) else None
                cap = (
                    latest_aj.get("per_tick_paid_forecast_cap_usd")
                    if isinstance(latest_aj, dict) else None
                )
                if isinstance(bp, str):
                    report.budget_profile = bp
                if isinstance(cap, (int, float)):
                    report.per_tick_paid_forecast_cap_usd = float(cap)

        # Phase 6D: derive ticks observed + avg tick spend from
        # tick_audit + forecasts.api_cost_usd. ``projected_14_day_spend``
        # extrapolates linearly assuming 96 ticks/day.
        ticks_observed = 0
        if _has_table(conn, "tick_audit"):
            ticks_observed = int(
                _scalar(
                    conn,
                    "SELECT count(*) FROM tick_audit WHERE status='COMPLETED'",
                    default=0,
                )
            )
        report.ticks_observed = ticks_observed
        forecast_spend_total = float(report.forecasts_spend_total_usd or 0.0)
        if ticks_observed > 0:
            report.avg_tick_paid_spend_usd = round(
                forecast_spend_total / ticks_observed, 6,
            )
            # Linear extrapolation: avg_per_tick * 96 ticks/day * 14 days.
            report.projected_14_day_spend_usd = round(
                report.avg_tick_paid_spend_usd * 96 * 14, 2,
            )

        return report
    finally:
        conn.close()


# --- rendering -------------------------------------------------------------


def render_funnel_report(report: FunnelReport) -> str:
    lines: list[str] = []
    lines.append(f"# Funnel report - {report.db_path}")
    if report.experiment_ids:
        lines.append(f"experiment_ids: {', '.join(report.experiment_ids)}")
    lines.append("")
    lines.append("## 1. Candidates loaded")
    lines.append(f"  total: {report.candidates_loaded}")
    lines.append("")
    lines.append("## 2. Universe filter")
    lines.append(f"  accepted: {report.universe_accepted}")
    lines.append(f"  rejected: {report.universe_rejected}")
    for reason, count in sorted(report.universe_reject_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    {reason}: {count}")
    if report.horizon_extended_count:
        lines.append(
            f"  horizon-extended shadow rows (30-90d would-pass): "
            f"{report.horizon_extended_count}"
        )
    lines.append("")
    lines.append("## 3. Selector (forecast_selection shadow rows)")
    lines.append(f"  selected: {report.selector_selected}")
    lines.append(f"  skipped : {report.selector_skipped}")
    for reason, count in sorted(report.selector_skip_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    {reason}: {count}")
    if report.exploration_promotions:
        preview = ", ".join(report.exploration_promotion_ids[:12])
        lines.append(
            f"  exploration_promotions: {report.exploration_promotions} ({preview})"
        )
    lines.append("")
    lines.append("## 4. Forecasts")
    lines.append(f"  total: {report.forecasts_total}  total_spend_usd: ${report.forecasts_spend_total_usd:.4f}")
    for tier, count in sorted(report.forecasts_by_model_tier.items(), key=lambda x: -x[1]):
        spend = report.forecasts_spend_by_model_tier.get(tier, 0.0)
        lines.append(f"    {tier}: {count}  spend=${spend:.4f}")
    lines.append("")
    lines.append("## 4.5 Web search (Phase 6B)")
    lines.append(
        f"  web_search_requests_total: {report.web_search_requests_total}"
    )
    lines.append(
        f"  cited_urls_total         : {report.cited_urls_total}"
    )
    lines.append(
        f"  evidence_quality_avg     : {report.evidence_quality_avg}"
    )
    lines.append(
        f"  web_sonnet_forecast count: {report.web_sonnet_forecast_count}"
    )
    lines.append(
        f"  web_opus_forecast count  : {report.web_opus_forecast_count}"
    )
    lines.append("")
    lines.append("## 5. Proposals")
    lines.append("  by edge_source:")
    for source, count in sorted(report.proposals_by_edge_source.items(), key=lambda x: -x[1]):
        lines.append(f"    {source}: {count}")
    lines.append("  by decision:")
    for decision, count in sorted(report.proposals_by_decision.items()):
        lines.append(f"    {decision}: {count}")
    if report.proposals_reject_reasons:
        lines.append("  reject reasons (excl. universe):")
        for reason, count in sorted(report.proposals_reject_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason}: {count}")
    lines.append(f"  accepted_proposals_count: {report.accepted_proposals_count}")
    lines.append("")
    lines.append("## 6. Intents + fills")
    lines.append(f"  intents_submitted: {report.intents_submitted}")
    if report.fills_by_status:
        for status, count in sorted(report.fills_by_status.items(), key=lambda x: -x[1]):
            lines.append(f"    fill_status={status}: {count}")
    else:
        lines.append("  (no fill rows)")
    # --- Phase 6R triage section -----------------------------------------
    lines.append("")
    lines.append("## 7. Phase 6R triage")
    lines.append(f"  fill_prob_below_skip rejects: {report.fill_prob_below_skip_count}")
    lines.append(
        f"  canary_fill_rescue_shadow rows : {report.canary_fill_rescue_shadow_count}"
    )
    lines.append(
        f"  canary_kelly_rescue_shadow rows: {report.canary_kelly_rescue_shadow_count}"
    )
    if report.top_fill_rescue_rows:
        lines.append("  top fill-rescue candidates (score desc):")
        for row in report.top_fill_rescue_rows[:5]:
            lines.append(
                "    "
                f"{row['market_id']} side={row.get('side','?')} "
                f"edge={row.get('edge_source','')} "
                f"score={row['score']:.4f} "
                f"original_fill_prob={row.get('original_fill_prob')} "
                f"canary25={row.get('canary_fill_prob_25')} "
                f"rescue_expected_fp={row.get('rescue_expected_fill_prob')} "
                f"age={row.get('quote_age_sec')}s"
            )
    if report.top_kelly_rescue_rows:
        lines.append("  top kelly-rescue candidates (score desc):")
        for row in report.top_kelly_rescue_rows[:5]:
            lines.append(
                "    "
                f"{row['market_id']} side={row.get('side','?')} "
                f"edge={row.get('edge_source','')} "
                f"rescue_edge_pp={row.get('rescue_edge_pp')} "
                f"canary25={row.get('canary_fill_prob_25')} "
                f"age={row.get('quote_age_sec')}s"
            )
    if report.quote_age_buckets:
        lines.append("  quote_age distribution (fill_prob_below_skip subset):")
        for bucket in ("<=60s", "60-180s", "180-300s", ">300s", "missing"):
            lines.append(f"    {bucket}: {report.quote_age_buckets.get(bucket, 0)}")
    # --- Phase 6D budget surface ---------------------------------------
    lines.append("")
    lines.append("## 8. Phase 6D budget-aware routing")
    lines.append(
        f"  budget_profile               : {report.budget_profile or '<unknown>'}"
    )
    lines.append(
        f"  per_tick_paid_forecast_cap   : "
        f"${report.per_tick_paid_forecast_cap_usd:.4f}"
        if report.per_tick_paid_forecast_cap_usd is not None
        else "  per_tick_paid_forecast_cap   : <unset>"
    )
    lines.append(
        f"  ticks_observed               : {report.ticks_observed}"
    )
    lines.append(
        f"  avg_tick_paid_spend          : ${report.avg_tick_paid_spend_usd:.4f}"
    )
    lines.append(
        f"  projected_14_day_spend       : ${report.projected_14_day_spend_usd:.2f}"
    )
    lines.append(
        f"  forecast_blocked_tick_budget : {report.forecast_blocked_tick_budget_count}"
    )
    lines.append(
        f"  opus_skipped_budget_count    : {report.opus_skipped_budget_count}"
    )
    if report.top_blocked_proposals:
        lines.append("  top blocked proposals (score desc):")
        for row in report.top_blocked_proposals[:5]:
            orig_fp = row.get("original_fill_prob")
            rescue_fp = row.get("rescue_expected_fill_prob")
            lines.append(
                "    "
                f"{row['market_id']}  score={row['score']:.4f}  "
                f"size=${row['size_usd']:.2f}  "
                f"fill_prob={row['fill_prob']:.3f}  "
                f"original_fill_prob={orig_fp}  "
                f"canary25={row.get('canary_fill_prob_25')}  "
                f"canary50={row.get('canary_fill_prob_50')}  "
                f"rescue_expected_fp={rescue_fp}  "
                f"age={row.get('quote_age_sec')}s  edge={row.get('edge_source')}"
            )
    return "\n".join(lines)


def report_to_dict(report: FunnelReport) -> dict[str, Any]:
    return asdict(report)


# --- CLI -------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.reports.funnel_report")
    parser.add_argument("path", help="experiment folder or path to state.sqlite3")
    parser.add_argument("--json", action="store_true", help="emit a JSON dump")
    parser.add_argument("--out", default=None, help="optional output file path")
    args = parser.parse_args(argv)
    report = build_funnel_report(args.path)
    if args.json:
        text = json.dumps(report_to_dict(report), indent=2, default=str)
    else:
        text = render_funnel_report(report)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
