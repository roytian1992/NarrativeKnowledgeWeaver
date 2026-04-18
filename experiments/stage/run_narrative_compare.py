from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.builder.narrative_graph_builder import NarrativeGraphBuilder
from core.utils.config import KAGConfig, _apply_global_locale_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run narrative-only compare on an existing KG workspace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--language", required=True, choices=["zh", "en"])
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--old-workspace", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text())


def _count_json_list(path: Path) -> int:
    data = _read_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    raise TypeError(f"Unsupported JSON shape for {path}")


def load_counts(workspace: Path) -> dict:
    ng = workspace / "narrative_graph"
    global_dir = ng / "global"
    episodes_dir = ng / "episodes"
    counts = {
        "episodes": _count_json_list(global_dir / "episodes.json"),
        "episode_support_edges": _count_json_list(global_dir / "episode_support_edges.json"),
        "episode_relations": _count_json_list(global_dir / "episode_relations.json"),
        "episode_relations_dag": _count_json_list(global_dir / "episode_relations_dag.json"),
        "storyline_candidates": _count_json_list(global_dir / "storyline_candidates.json"),
        "storylines": _count_json_list(global_dir / "storylines.json"),
        "storyline_support_edges": _count_json_list(global_dir / "storyline_support_edges.json"),
        "storyline_relations": _count_json_list(global_dir / "storyline_relations.json"),
        "candidate_pairs": _count_json_list(episodes_dir / "candidate_pairs.json"),
    }
    cycle_log = _read_json(global_dir / "episode_cycle_break_saber_log.json")
    if isinstance(cycle_log, dict):
        counts["cycle_break_in_edges"] = int(cycle_log.get("in_edges", 0))
        counts["cycle_break_out_edges"] = int(cycle_log.get("out_edges", 0))
    else:
        counts["cycle_break_in_edges"] = 0
        counts["cycle_break_out_edges"] = 0
    return counts


def build_cfg(args: argparse.Namespace, workspace: Path) -> KAGConfig:
    cfg = copy.deepcopy(KAGConfig.from_yaml(args.config))
    for global_cfg in (cfg.global_, cfg.global_config):
        global_cfg.language = args.language
        global_cfg.locale = args.language
        global_cfg.doc_type = "screenplay"
        global_cfg.aggregation_mode = "narrative"
        _apply_global_locale_paths(global_cfg)

    workers = max(1, int(args.workers))
    cfg.knowledge_graph_builder.file_path = str(workspace / "knowledge_graph")
    cfg.narrative_graph_builder.file_path = str(workspace / "narrative_graph")
    cfg.storage.graph_store_path = str(workspace / "knowledge_graph" / "graph_runtime_langgraph.pkl")
    cfg.storage.vector_store_path = str(workspace / "vector_store")
    cfg.storage.sql_database_path = str(workspace / "sql")
    cfg.narrative_graph_builder.max_workers = workers
    return cfg


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    workspace = run_root / "workspace"
    report_path = run_root / "narrative_report.json"

    cfg = build_cfg(args, workspace)
    old_counts = load_counts(Path(args.old_workspace).resolve())

    print(
        json.dumps(
            {
                "movie_id": args.movie_id,
                "language": args.language,
                "run_root": str(run_root),
                "workspace": str(workspace),
                "workers": args.workers,
                "narrative_knobs": {
                    "episode_relation_similarity_threshold": cfg.narrative_graph_builder.episode_relation_similarity_threshold,
                    "episode_relation_primary_anchor_fields": cfg.narrative_graph_builder.episode_relation_primary_anchor_fields,
                    "episode_relation_context_anchor_fields": cfg.narrative_graph_builder.episode_relation_context_anchor_fields,
                    "episode_relation_max_primary_bucket_size": cfg.narrative_graph_builder.episode_relation_max_primary_bucket_size,
                    "episode_relation_max_context_bucket_size": cfg.narrative_graph_builder.episode_relation_max_context_bucket_size,
                    "episode_relation_context_requires_primary_pair": cfg.narrative_graph_builder.episode_relation_context_requires_primary_pair,
                    "episode_relation_topk_per_episode": cfg.narrative_graph_builder.episode_relation_topk_per_episode,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    timings = {}
    total_start = time.time()
    builder = NarrativeGraphBuilder(cfg)
    try:
        stage_start = time.time()
        builder.extract_episodes(
            limit_documents=200000,
            document_concurrency=max(1, int(args.workers)),
            store_episode_support_edges=True,
            ensure_episode_embeddings=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        timings["extract_episodes_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_episode_relations(
            episode_pair_concurrency=max(1, int(args.workers)),
            max_episode_pairs_global=200000,
            cross_document_only=False,
            similarity_threshold=float(
                getattr(cfg.narrative_graph_builder, "episode_relation_similarity_threshold", 0.55) or 0.55
            ),
            ensure_episode_embeddings=True,
            show_pair_progress=False,
            save_pair_json=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        timings["extract_episode_relations_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.break_episode_cycles(method="saber")
        timings["break_episode_cycles_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.build_storyline_candidates(method="trie", min_trunk_len=2)
        timings["build_storyline_candidates_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_storylines_from_candidates(
            ensure_storyline_embeddings=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        timings["extract_storylines_from_candidates_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.extract_storyline_relations(
            max_storyline_pairs_global=500,
            similarity_threshold=0.5,
            overlap_pair_only=True,
            min_shared_anchor_count=2,
            show_pair_progress=False,
            storyline_pair_concurrency=max(1, int(args.workers)),
        )
        timings["extract_storyline_relations_sec"] = round(time.time() - stage_start, 3)

        stage_start = time.time()
        builder.load_json_to_graph_store(
            store_episodes=True,
            store_support_edges=True,
            store_episode_relations=True,
            store_storylines=True,
            store_storyline_support_edges=True,
            store_storyline_relations=True,
        )
        timings["load_json_to_graph_store_sec"] = round(time.time() - stage_start, 3)
    finally:
        try:
            builder.graph_store.close()
        except Exception:
            pass

    timings["total_sec"] = round(time.time() - total_start, 3)
    new_counts = load_counts(workspace)
    delta = {key: new_counts.get(key, 0) - old_counts.get(key, 0) for key in sorted(set(old_counts) | set(new_counts))}

    report = {
        "movie_id": args.movie_id,
        "language": args.language,
        "mode": "narrative_only_compare",
        "config_path": str(Path(args.config).resolve()),
        "workers": args.workers,
        "paths": {
            "old_workspace": str(Path(args.old_workspace).resolve()),
            "run_root": str(run_root),
            "workspace": str(workspace),
        },
        "old_counts": old_counts,
        "new_counts": new_counts,
        "delta": delta,
        "timings_sec": timings,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT_WRITTEN {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
