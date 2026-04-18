from __future__ import annotations

import argparse
import copy
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.builder.graph_builder import KnowledgeGraphBuilder
from core.utils.config import KAGConfig, _apply_global_locale_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild one movie's KG in a fresh workspace and compare counts.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--language", required=True, choices=["zh", "en"], help="Movie language.")
    parser.add_argument("--json-file", required=True, help="Path to converted script json.")
    parser.add_argument("--old-workspace", required=True, help="Path to old workspace for baseline counts.")
    parser.add_argument("--run-root", required=True, help="Output directory for the new comparison run.")
    parser.add_argument("--movie-id", required=True, help="Movie id for reporting.")
    parser.add_argument("--workers", type=int, default=32, help="KG build worker count.")
    return parser.parse_args()


def load_counts(workspace: Path) -> dict:
    kg = workspace / "knowledge_graph"
    out = {
        "docs": len(json.loads((kg / "doc2chunks.json").read_text())),
        "chunks": len(json.loads((kg / "all_document_chunks.json").read_text())),
        "entities_basic": len(json.loads((kg / "entity_basic_info.json").read_text())),
        "relations_basic": len(json.loads((kg / "relation_basic_info.json").read_text())),
        "entities_refined": len(json.loads((kg / "entity_info_refined.json").read_text())),
        "relations_refined": len(json.loads((kg / "relation_info_refined.json").read_text())),
        "doc_entities": len(json.loads((kg / "doc_entities.json").read_text())),
        "doc_entity_edges": len(json.loads((kg / "doc_entity_edges.json").read_text())),
        "interactions": len(json.loads((workspace / "interactions" / "interaction_records_list.json").read_text())),
    }
    with open(kg / "graph_nx.pkl", "rb") as f:
        graph = pickle.load(f)
    out["graph_nodes"] = int(graph.number_of_nodes())
    out["graph_edges"] = int(graph.number_of_edges())
    conn = sqlite3.connect(workspace / "sql" / "Interaction.db")
    try:
        cur = conn.cursor()
        cur.execute("select count(*) from Interaction_info")
        out["interaction_rows"] = int(cur.fetchone()[0])
    finally:
        conn.close()
    return out


def build_cfg(args: argparse.Namespace, workspace: Path) -> KAGConfig:
    cfg = copy.deepcopy(KAGConfig.from_yaml(args.config))
    for global_cfg in (cfg.global_, cfg.global_config):
        global_cfg.language = args.language
        global_cfg.locale = args.language
        global_cfg.doc_type = "screenplay"
        global_cfg.aggregation_mode = "narrative"
        _apply_global_locale_paths(global_cfg)

    workers = max(1, int(args.workers))
    cfg.document_processing.max_workers = workers
    cfg.document_processing.reset_vector_collections = True
    cfg.knowledge_graph_builder.max_workers = workers
    cfg.knowledge_graph_builder.file_path = str(workspace / "knowledge_graph")
    cfg.storage.graph_store_path = str(workspace / "knowledge_graph" / "graph_runtime_langgraph.pkl")
    cfg.storage.vector_store_path = str(workspace / "vector_store")
    cfg.storage.sql_database_path = str(workspace / "sql")
    cfg.extraction_memory.enabled = False
    return cfg


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    workspace = run_root / "workspace"
    interaction_dir = workspace / "interactions"
    sql_dir = workspace / "sql"
    interaction_json_path = interaction_dir / "interaction_results.json"
    interaction_list_json_path = interaction_dir / "interaction_records_list.json"
    sql_db_path = sql_dir / "Interaction.db"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    sql_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args, workspace)
    baseline_counts = load_counts(Path(args.old_workspace).resolve())
    timings = {}

    print(
        json.dumps(
            {
                "movie_id": args.movie_id,
                "language": args.language,
                "json_file": str(Path(args.json_file).resolve()),
                "old_workspace": str(Path(args.old_workspace).resolve()),
                "run_root": str(run_root),
                "workers": args.workers,
                "kg_knobs": {
                    "extraction_pack_short_chunk_word_threshold": cfg.knowledge_graph_builder.extraction_pack_short_chunk_word_threshold,
                    "extraction_pack_max_words": cfg.knowledge_graph_builder.extraction_pack_max_words,
                    "property_context_max_edge_descriptions": cfg.knowledge_graph_builder.property_context_max_edge_descriptions,
                    "property_context_max_total_words": cfg.knowledge_graph_builder.property_context_max_total_words,
                    "property_context_dedupe_descriptions": cfg.knowledge_graph_builder.property_context_dedupe_descriptions,
                    "interaction_min_entity_candidates": cfg.knowledge_graph_builder.interaction_min_entity_candidates,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    total_start = time.time()
    builder = KnowledgeGraphBuilder(cfg, use_memory=False)
    try:
        stage_start = time.time()
        builder.prepare_chunks(
            json_file_path=str(Path(args.json_file).resolve()),
            reset_output_dir=True,
            reset_vector_collections=True,
        )
        timings["prepare_chunks_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_entity_and_relation(
            retries=3,
            concurrency=max(1, int(args.workers)),
            per_task_timeout=int(cfg.knowledge_graph_builder.per_task_timeout or 2400),
            reset_outputs=True,
        )
        timings["extract_entity_and_relation_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.run_extraction_refinement()
        timings["run_extraction_refinement_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.build_entity_and_relation_basic_info()
        timings["build_entity_and_relation_basic_info_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.postprocess_and_save()
        timings["postprocess_and_save_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_properties()
        timings["extract_properties_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_interactions(
            retries=3,
            concurrency=max(1, int(args.workers)),
            per_task_timeout=int(cfg.knowledge_graph_builder.per_task_timeout or 2400),
            interaction_json_path=str(interaction_json_path),
            interaction_list_json_path=str(interaction_list_json_path),
        )
        timings["extract_interactions_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.store_interactions_to_sql(
            interaction_list_json_path=str(interaction_list_json_path),
            sql_db_path=str(sql_db_path),
            reset_database=True,
            reset_table=True,
        )
        timings["store_interactions_to_sql_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.build_doc_entities()
        timings["build_doc_entities_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.load_json_to_graph_store()
        timings["load_json_to_graph_store_sec"] = round(time.time() - stage_start, 3)
    finally:
        try:
            builder.graph_store.close()
        except Exception:
            pass

    timings["total_sec"] = round(time.time() - total_start, 3)
    new_counts = load_counts(workspace)
    delta = {key: new_counts.get(key, 0) - baseline_counts.get(key, 0) for key in sorted(set(baseline_counts) | set(new_counts))}

    report = {
        "movie_id": args.movie_id,
        "language": args.language,
        "mode": "kg_only_compare",
        "config_path": str(Path(args.config).resolve()),
        "workers": args.workers,
        "paths": {
            "json_file": str(Path(args.json_file).resolve()),
            "old_workspace": str(Path(args.old_workspace).resolve()),
            "run_root": str(run_root),
            "workspace": str(workspace),
        },
        "kg_knobs": {
            "extraction_pack_short_chunk_word_threshold": cfg.knowledge_graph_builder.extraction_pack_short_chunk_word_threshold,
            "extraction_pack_max_words": cfg.knowledge_graph_builder.extraction_pack_max_words,
            "property_context_max_edge_descriptions": cfg.knowledge_graph_builder.property_context_max_edge_descriptions,
            "property_context_max_total_words": cfg.knowledge_graph_builder.property_context_max_total_words,
            "property_context_dedupe_descriptions": cfg.knowledge_graph_builder.property_context_dedupe_descriptions,
            "interaction_min_entity_candidates": cfg.knowledge_graph_builder.interaction_min_entity_candidates,
        },
        "old_counts": baseline_counts,
        "new_counts": new_counts,
        "delta": delta,
        "timings_sec": timings,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT_WRITTEN {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
