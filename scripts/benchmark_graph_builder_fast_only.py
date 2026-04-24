from __future__ import annotations

import argparse
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
    coverage_repair_triggered_chunks = 0
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
            coverage_repair_triggered_chunks += int(stats.get("coverage_repair_triggered_chunks", 0) or 0)
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
        "coverage_repair_triggered_chunks": coverage_repair_triggered_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph_builder fast extraction stage only.")
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

    kg_dir = output_dir / "knowledge_graph"
    _ensure_workspace_inputs(workspace_dir, kg_dir)

    config = load_config(str(Path(args.config).resolve()))
    config.knowledge_graph_builder.file_path = str(kg_dir)
    config.knowledge_graph_builder.max_workers = int(args.workers)
    config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"
    config.llm.timeout = int(args.llm_timeout)
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    if args.llm_model_name:
        config.llm.model_name = args.llm_model_name

    builder = KnowledgeGraphBuilder(config)

    start = time.perf_counter()
    builder.extract_entity_and_relation(
        verbose=True,
        retries=3,
        per_task_timeout=float(args.llm_timeout) * 4,
        concurrency=int(args.workers),
        reset_outputs=True,
        aggressive_clean=True,
        pipeline_mode="event_first_fast",
        short_scene_skip_word_threshold=int(args.short_scene_skip_word_threshold),
    )
    elapsed = time.perf_counter() - start

    summary = _summarize_extraction_results(kg_dir / "extraction_results.json")
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["pipeline_mode"] = "event_first_fast"
    summary["relation_extraction_mode"] = "open_then_ground"
    summary["workspace_dir"] = str(workspace_dir)
    summary["workers"] = int(args.workers)

    _dump_json(output_dir / "benchmark_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
