from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_old_no_strategy(paths: List[str], article_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    article_id_set = set(article_ids)
    for path in paths:
        data = _load_json(path)
        for article_id, pack in (data.get("article_summary") or {}).items():
            if article_id in article_id_set:
                out[article_id] = dict((pack.get("no_strategy_agent") or {}))
    return out


def _collect_setting(path: str, setting_name: str, article_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    article_id_set = set(article_ids)
    for article_id, pack in (data.get("article_summary") or {}).items():
        if article_id in article_id_set:
            out[article_id] = dict((pack.get(setting_name) or {}))
    return out


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--new-report", required=True)
    parser.add_argument("--hybrid-report", required=True)
    parser.add_argument("--old-report", action="append", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    article_ids = [item["article_name"] for item in (manifest.get("eval") or [])]

    old_rows = _collect_old_no_strategy(args.old_report, article_ids)
    hybrid_rows = _collect_setting(args.hybrid_report, "traditional_hybrid_rag_bm25", article_ids)
    new_rows = _collect_setting(args.new_report, "no_strategy_agent", article_ids)

    comparison: List[Dict[str, Any]] = []
    for article_id in article_ids:
        old = old_rows.get(article_id, {})
        hybrid = hybrid_rows.get(article_id, {})
        new = new_rows.get(article_id, {})
        comparison.append(
            {
                "article_id": article_id,
                "old_pass_accuracy": _safe_float(old.get("pass_accuracy")),
                "old_overall_accuracy": _safe_float(old.get("overall_accuracy")),
                "hybrid_accuracy": _safe_float(hybrid.get("overall_accuracy")),
                "new_pass_accuracy": _safe_float(new.get("pass_accuracy")),
                "new_overall_accuracy": _safe_float(new.get("overall_accuracy")),
                "delta_vs_old_pass": _safe_float(new.get("pass_accuracy")) - _safe_float(old.get("pass_accuracy")),
                "delta_vs_old_overall": _safe_float(new.get("overall_accuracy")) - _safe_float(old.get("overall_accuracy")),
                "delta_vs_hybrid": _safe_float(new.get("overall_accuracy")) - _safe_float(hybrid.get("overall_accuracy")),
            }
        )

    summary = {
        "article_count": len(comparison),
        "old_pass_accuracy_avg": round(mean(row["old_pass_accuracy"] for row in comparison), 4) if comparison else 0.0,
        "old_overall_accuracy_avg": round(mean(row["old_overall_accuracy"] for row in comparison), 4) if comparison else 0.0,
        "hybrid_accuracy_avg": round(mean(row["hybrid_accuracy"] for row in comparison), 4) if comparison else 0.0,
        "new_pass_accuracy_avg": round(mean(row["new_pass_accuracy"] for row in comparison), 4) if comparison else 0.0,
        "new_overall_accuracy_avg": round(mean(row["new_overall_accuracy"] for row in comparison), 4) if comparison else 0.0,
        "avg_delta_vs_old_pass": round(mean(row["delta_vs_old_pass"] for row in comparison), 4) if comparison else 0.0,
        "avg_delta_vs_old_overall": round(mean(row["delta_vs_old_overall"] for row in comparison), 4) if comparison else 0.0,
        "avg_delta_vs_hybrid": round(mean(row["delta_vs_hybrid"] for row in comparison), 4) if comparison else 0.0,
    }

    payload = {
        "summary": summary,
        "comparison": comparison,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    output_json = str(args.output_json or "").strip()
    if output_json:
        Path(output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
