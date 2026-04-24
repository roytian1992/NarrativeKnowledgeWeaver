from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.builder.graph_builder import KnowledgeGraphBuilder
from core.utils.config import load_config
from scripts.compare_relation_modes_on_workspace import _dump_json, _load_json


def _ensure_workspace_inputs(src_workspace_dir: Path, dst_kg_dir: Path) -> None:
    dst_kg_dir.mkdir(parents=True, exist_ok=True)
    for name in ["doc2chunks.json", "doc2chunks_index.json", "all_document_chunks.json"]:
        src = src_workspace_dir / "knowledge_graph" / name
        if src.exists():
            shutil.copy2(src, dst_kg_dir / name)


def _summarize_extraction_results(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    if not isinstance(data, dict):
        return {}
    total_docs = len(data)
    ok_docs = 0
    failed_docs = 0
    skipped_docs = 0
    entity_total = 0
    relation_total = 0
    packed_chunk_total = 0
    word_total = 0
    auto_rel_total = 0
    open_rel_total = 0
    for payload in data.values():
        if not isinstance(payload, dict):
            continue
        if payload.get("ok"):
            ok_docs += 1
        else:
            failed_docs += 1
        if payload.get("skipped_extraction"):
            skipped_docs += 1
        entity_total += len(payload.get("entities") or [])
        relation_total += len(payload.get("relations") or [])
        stats = payload.get("stats") or {}
        if isinstance(stats, dict):
            packed_chunk_total += int(stats.get("packed_chunk_count", 0) or 0)
            word_total += int(stats.get("word_count", 0) or 0)
            auto_rel_total += int(stats.get("auto_relation_count", 0) or 0)
            open_rel_total += int(stats.get("open_relation_count_after_fix", 0) or 0)
    return {
        "total_docs": total_docs,
        "ok_docs": ok_docs,
        "failed_docs": failed_docs,
        "skipped_docs": skipped_docs,
        "entity_total": entity_total,
        "relation_total": relation_total,
        "packed_chunk_total": packed_chunk_total,
        "word_total": word_total,
        "auto_relation_total": auto_rel_total,
        "open_relation_after_fix_total": open_rel_total,
    }


def _run_one(
    *,
    src_workspace_dir: Path,
    output_root: Path,
    config_path: Path,
    llm_timeout: int,
    llm_base_url: str,
    llm_model_name: str,
    workers: int,
    pipeline_mode: str,
    relation_mode: str,
    short_scene_skip_word_threshold: int,
) -> Dict[str, Any]:
    kg_dir = output_root / "knowledge_graph"
    _ensure_workspace_inputs(src_workspace_dir, kg_dir)

    config = load_config(str(config_path))
    config.knowledge_graph_builder.file_path = str(kg_dir)
    config.knowledge_graph_builder.max_workers = int(workers)
    config.knowledge_graph_builder.relation_extraction_mode = relation_mode
    config.llm.timeout = int(llm_timeout)
    if llm_base_url:
        config.llm.base_url = llm_base_url
    if llm_model_name:
        config.llm.model_name = llm_model_name

    builder = KnowledgeGraphBuilder(config)

    start = time.perf_counter()
    builder.extract_entity_and_relation(
        verbose=True,
        retries=3,
        per_task_timeout=float(llm_timeout) * 4,
        concurrency=int(workers),
        reset_outputs=True,
        aggressive_clean=True,
        pipeline_mode=pipeline_mode,
        short_scene_skip_word_threshold=short_scene_skip_word_threshold,
    )
    elapsed = time.perf_counter() - start

    summary = _summarize_extraction_results(kg_dir / "extraction_results.json")
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["pipeline_mode"] = pipeline_mode
    summary["relation_extraction_mode"] = relation_mode
    return summary


def _render_report(*, workspace_dir: Path, old_summary: Dict[str, Any], fast_summary: Dict[str, Any]) -> str:
    old_time = float(old_summary.get("elapsed_sec", 0.0) or 0.0)
    fast_time = float(fast_summary.get("elapsed_sec", 0.0) or 0.0)
    saved = old_time - fast_time
    pct = (saved / old_time * 100.0) if old_time > 0 else 0.0
    lines = [
        "# Graph Builder Extraction Benchmark",
        "",
        f"- source_workspace: `{workspace_dir}`",
        "",
        "## Runtime",
        "",
        f"- legacy elapsed_sec: {old_time:.3f}",
        f"- fast elapsed_sec: {fast_time:.3f}",
        f"- saved_sec: {saved:.3f}",
        f"- saved_pct: {pct:.3f}",
        "",
        "## Output Counts",
        "",
        "| mode | docs | skipped_docs | entities | relations | auto_rel | open_rel_after_fix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| legacy | {int(old_summary.get('total_docs', 0))} | {int(old_summary.get('skipped_docs', 0))} | "
            f"{int(old_summary.get('entity_total', 0))} | {int(old_summary.get('relation_total', 0))} | "
            f"{int(old_summary.get('auto_relation_total', 0))} | {int(old_summary.get('open_relation_after_fix_total', 0))} |"
        ),
        (
            f"| fast | {int(fast_summary.get('total_docs', 0))} | {int(fast_summary.get('skipped_docs', 0))} | "
            f"{int(fast_summary.get('entity_total', 0))} | {int(fast_summary.get('relation_total', 0))} | "
            f"{int(fast_summary.get('auto_relation_total', 0))} | {int(fast_summary.get('open_relation_after_fix_total', 0))} |"
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph_builder extraction stage: legacy vs fast.")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model-name", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--short-scene-skip-word-threshold", type=int, default=25)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_summary = _run_one(
        src_workspace_dir=workspace_dir,
        output_root=output_dir / "legacy_run",
        config_path=Path(args.config).resolve(),
        llm_timeout=int(args.llm_timeout),
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_model_name,
        workers=int(args.workers),
        pipeline_mode="legacy",
        relation_mode="schema_direct",
        short_scene_skip_word_threshold=0,
    )
    fast_summary = _run_one(
        src_workspace_dir=workspace_dir,
        output_root=output_dir / "fast_run",
        config_path=Path(args.config).resolve(),
        llm_timeout=int(args.llm_timeout),
        llm_base_url=args.llm_base_url,
        llm_model_name=args.llm_model_name,
        workers=int(args.workers),
        pipeline_mode="event_first_fast",
        relation_mode="open_then_ground",
        short_scene_skip_word_threshold=int(args.short_scene_skip_word_threshold),
    )

    result = {
        "workspace_dir": str(workspace_dir),
        "workers": int(args.workers),
        "legacy": old_summary,
        "fast": fast_summary,
    }
    legacy_t = float(old_summary.get("elapsed_sec", 0.0) or 0.0)
    fast_t = float(fast_summary.get("elapsed_sec", 0.0) or 0.0)
    saved = legacy_t - fast_t
    result["comparison"] = {
        "saved_sec": round(saved, 3),
        "saved_pct": round((saved / legacy_t * 100.0), 3) if legacy_t > 0 else 0.0,
    }

    _dump_json(output_dir / "benchmark_summary.json", result)
    (output_dir / "benchmark_report.md").write_text(
        _render_report(workspace_dir=workspace_dir, old_summary=old_summary, fast_summary=fast_summary),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
