from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.quality.run_quality_benchmark import (
    QUALITY_SETTING_ORDER,
    KAGConfig,
    StrategyLibraryAccumulator,
    _build_article_result_payload,
    _evaluate_online_setting_incremental,
    _load_article_qas,
    _load_manifest,
    _set_global_language,
    _summarize_setting,
    _write_article_result_files,
    load_existing_article_workspace_or_raise,
)
from core.utils.general_utils import json_dump_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one online QUALITY setting from an existing article workspace.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--manifest", default="experiments/quality/artifacts/split_manifest.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--article-name", required=True)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--setting-name", default="online_strategy_agent")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-max-workers", type=int, default=4)
    parser.add_argument("--online-warmup-questions", type=int, default=5)
    parser.add_argument("--online-batch-size", type=int, default=3)
    parser.add_argument("--self-bootstrap-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--self-bootstrap-max-questions", type=int, default=1)
    parser.add_argument("--disable-sql-tools", action="store_true")
    parser.add_argument("--subagent", action="store_true")
    args = parser.parse_args()
    enable_sql_tools = not bool(args.disable_sql_tools)

    base_cfg = KAGConfig.from_yaml(args.config)
    _set_global_language(base_cfg, "en")
    base_cfg.global_.doc_type = "general"
    base_cfg.global_config.doc_type = "general"
    base_cfg.global_.aggregation_mode = "narrative"
    base_cfg.global_config.aggregation_mode = "narrative"

    manifest = _load_manifest((REPO_ROOT / args.manifest).resolve())
    article = None
    for row in manifest.get("eval") or []:
        if str(row.get("article_name", "") or "").strip() == str(args.article_name or "").strip():
            article = row
            break
    if article is None:
        raise RuntimeError(f"Article not found in manifest: {args.article_name}")

    article_name = str(article.get("article_name", "") or "").strip()
    article_qa_path = Path(str(article.get("qa_path", "") or ""))
    workspace_dir = Path(str(args.workspace_dir or "")).resolve()
    qa_rows = _load_article_qas(article_qa_path, with_answer=True)

    run_root = REPO_ROOT / "experiments" / "quality" / "runs" / str(args.run_name or "").strip()
    report_root = run_root / "reports"
    runtime_root = run_root / "runtime"
    report_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    online_runtime_library = runtime_root / "online" / "strategy_library.json"
    online_runtime_tool_meta = runtime_root / "online" / "tool_metadata"
    online_accumulator = StrategyLibraryAccumulator(
        cfg=base_cfg,
        runtime_library_path=online_runtime_library,
        runtime_tool_metadata_dir=online_runtime_tool_meta,
        output_dir=report_root / "online_library_build",
        dataset_name=f"{args.run_name}_online",
    )
    online_accumulator.clear_runtime()

    bootstrap_mode = str(args.self_bootstrap_mode or "auto").strip().lower()
    resolved_self_bootstrap_max_questions = max(0, int(args.self_bootstrap_max_questions or 0))
    if bootstrap_mode == "off":
        resolved_self_bootstrap_max_questions = 0
    elif bootstrap_mode == "on":
        resolved_self_bootstrap_max_questions = max(1, resolved_self_bootstrap_max_questions or 1)

    # Strictly reuse an existing extracted workspace. This must never trigger rebuild.
    load_existing_article_workspace_or_raise(
        base_cfg=base_cfg,
        workspace_dir=workspace_dir,
        strategy_enabled=False,
        subagent_enabled=False,
    )

    setting_results, online_training_reports = _evaluate_online_setting_incremental(
        base_cfg=base_cfg,
        run_root=run_root,
        report_root=report_root,
        workspace_dir=workspace_dir,
        article_name=article_name,
        qa_rows=qa_rows,
        repeats=max(1, int(args.repeats or 1)),
        setting_name=str(args.setting_name or "online_strategy_agent").strip(),
        subagent_enabled=bool(args.subagent),
        online_accumulator=online_accumulator,
        self_bootstrap_max_questions=resolved_self_bootstrap_max_questions,
        warmup_questions=max(0, int(args.online_warmup_questions or 0)),
        batch_size=max(1, int(args.online_batch_size or 1)),
        eval_max_workers=max(1, int(args.eval_max_workers or 1)),
        enable_sql_tools=enable_sql_tools,
    )

    final_online_summary = online_accumulator.export(consolidate=True, export_name="final")
    article_payload = _build_article_result_payload(
        article_name=article_name,
        setting_results={
            setting: (setting_results if setting == str(args.setting_name).strip() else [])
            for setting in QUALITY_SETTING_ORDER
        },
        online_training_reports=online_training_reports,
        repeats=max(1, int(args.repeats or 1)),
    )
    _write_article_result_files(report_root=report_root, payload=article_payload)

    summary = _summarize_setting(
        setting_results,
        question_count=len({str(item.get("question_id", "") or "") for item in setting_results}),
        repeats=max(1, int(args.repeats or 1)),
    )
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": str(args.run_name or "").strip(),
        "article_name": article_name,
        "workspace_dir": str(workspace_dir),
        "setting": str(args.setting_name or "").strip(),
        "repeats": max(1, int(args.repeats or 1)),
        "self_bootstrap_mode": bootstrap_mode,
        "self_bootstrap_max_questions": resolved_self_bootstrap_max_questions,
        "enable_sql_tools": enable_sql_tools,
        "question_count": len(qa_rows),
        "summary": summary,
        "online_library_summary": final_online_summary,
        "online_training_report_count": len(online_training_reports),
    }
    out_path = report_root / "single_online_setting_result.json"
    json_dump_atomic(str(out_path), result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
