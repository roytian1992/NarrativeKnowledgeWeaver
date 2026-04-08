from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.functions.memory_management.strategy_query_pattern import StrategyQueryPatternExtractor
from core.strategy_training.strategy_cluster_manager import StrategyTemplateClusterManager
from core.strategy_training.strategy_runtime_assets import StrategyRuntimeAssetManager
from core.model_providers.openai_llm import OpenAILLM
from core.storage.vector_store import VectorStore
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic
from core.utils.prompt_loader import YAMLPromptLoader


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_template(template: Dict[str, Any], query_pattern: Dict[str, Any], embedding_model: Any) -> Dict[str, Any]:
    pattern_text = StrategyQueryPatternExtractor.pattern_to_text(query_pattern)
    embedding = None
    if embedding_model is not None and hasattr(embedding_model, "embed_query"):
        try:
            embedding = [float(x) for x in embedding_model.embed_query(pattern_text)]
        except Exception:
            embedding = None
    out = dict(template or {})
    out["query_pattern"] = dict(query_pattern or {})
    out["query_abstract"] = str(query_pattern.get("query_abstract", "") or "")
    out["query_pattern_text"] = pattern_text
    if embedding is not None:
        out["pattern_embedding"] = embedding
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recluster strategy memory from existing per-question artifacts.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-dir-name", default="recluster_from_artifacts")
    parser.add_argument("--runtime-library-path", default=None)
    args = parser.parse_args()

    cfg = KAGConfig.from_yaml(args.config)
    training_root = (PROJECT_ROOT / str(args.training_root)).resolve()
    output_root = training_root / str(args.output_dir_name or "recluster_from_artifacts")
    output_root.mkdir(parents=True, exist_ok=True)

    detail_dir = training_root / "distilled" / "per_question"
    if not detail_dir.exists():
        raise FileNotFoundError(f"Missing per-question directory: {detail_dir}")

    runtime_library_path = (
        Path(args.runtime_library_path).resolve()
        if args.runtime_library_path
        else (PROJECT_ROOT / str(getattr(cfg.strategy_memory, "library_path", "data/memory/strategy/strategy_library.json"))).resolve()
    )
    runtime_tool_metadata_dir = (
        PROJECT_ROOT / str(getattr(cfg.strategy_memory, "tool_metadata_runtime_dir", "data/memory/strategy/tool_metadata"))
    ).resolve()

    prompt_loader = YAMLPromptLoader(cfg.global_config.prompt_dir)
    llm = OpenAILLM(cfg)
    embedding_store = VectorStore(cfg, "document")
    embedding_model = getattr(embedding_store, "embedding_model", None)
    cluster_manager = StrategyTemplateClusterManager(
        prompt_loader=prompt_loader,
        llm=llm,
        embedding_model=embedding_model,
        candidate_top_k=getattr(cfg.strategy_memory, "merge_candidate_top_k", 3),
        min_candidate_score=getattr(cfg.strategy_memory, "merge_min_candidate_score", 0.28),
        consolidation_rounds=getattr(cfg.strategy_memory, "consolidation_rounds", 1),
        max_members_for_distill_prompt=getattr(cfg.strategy_memory, "cluster_distill_max_members", 12),
    )
    runtime_asset_manager = StrategyRuntimeAssetManager(
        library_path=str(runtime_library_path),
        tool_metadata_runtime_dir=str(runtime_tool_metadata_dir),
    )

    question_summaries: List[Dict[str, Any]] = []
    failed_summaries: List[Dict[str, Any]] = []

    detail_paths = sorted(detail_dir.glob("q*.json"), key=lambda p: int(p.stem.lstrip("q") or 0))
    for path in detail_paths:
        result = read_json(path, {})
        if not isinstance(result, dict):
            continue
        if bool(result.get("successful")):
            query_pattern = result.get("query_pattern") if isinstance(result.get("query_pattern"), dict) else {}
            template_seed = result.get("template_seed") if isinstance(result.get("template_seed"), dict) else {}
            raw_template = embed_template(template_seed, query_pattern, embedding_model)
            cluster_id = cluster_manager.add_template(raw_template)
            attempts = list(result.get("attempts") or [])
            question_summaries.append(
                {
                    "question_id": str(result.get("question_id", "") or ""),
                    "question": str(result.get("question", "") or ""),
                    "successful_attempt_count": sum(1 for item in attempts if bool((item.get("judge") or {}).get("is_correct", False))),
                    "best_attempt_id": str(result.get("best_attempt_id", "") or ""),
                    "best_retry_index": int(result.get("best_retry_index", 0) or 0),
                    "best_retry_instruction": str(result.get("best_retry_instruction", "") or ""),
                    "best_effective_tool_chain": list(result.get("best_effective_tool_chain") or []),
                    "best_raw_tool_chain": list(result.get("best_raw_tool_chain") or []),
                    "best_judge": result.get("best_judge", {}),
                    "query_pattern": query_pattern,
                    "cluster_id": cluster_id,
                    "template": raw_template,
                }
            )
        else:
            failure = result.get("failure") if isinstance(result.get("failure"), dict) else {}
            failed_summaries.append(
                {
                    "question_id": str(failure.get("question_id", result.get("question_id", "")) or ""),
                    "question": str(failure.get("question", result.get("question", "")) or ""),
                    "failure_summary": str(failure.get("failure_summary", "") or ""),
                    "likely_causes": list(failure.get("likely_causes") or []),
                    "recommended_improvements": list(failure.get("recommended_improvements") or []),
                }
            )

    consolidation_merge_count = cluster_manager.consolidate()
    cluster_payload = cluster_manager.export_training_payload()
    raw_templates = list(cluster_manager.raw_templates)
    runtime_templates = cluster_manager.runtime_templates()

    template_to_cluster: Dict[str, str] = {}
    for cluster in cluster_payload.get("clusters", []):
        cluster_id = str(cluster.get("cluster_id", "") or "")
        for template_id in cluster.get("member_template_ids", []) or []:
            template_to_cluster[str(template_id or "")] = cluster_id

    for item in question_summaries:
        template_id = str((item.get("template") or {}).get("template_id", "") or "")
        if template_id and template_id in template_to_cluster:
            item["cluster_id"] = template_to_cluster[template_id]

    template_source_index = {
        str(cluster.get("cluster_id", "") or ""): {
            "template_id": str(cluster.get("cluster_id", "") or ""),
            "pattern_name": str(cluster.get("pattern_name", "") or ""),
            "source_question_ids": list(cluster.get("source_question_ids") or []),
            "source_questions": list(cluster.get("source_questions") or []),
            "member_template_ids": list(cluster.get("member_template_ids") or []),
        }
        for cluster in cluster_payload.get("clusters", [])
        if str(cluster.get("cluster_id", "") or "")
    }

    tool_description_candidate_path = training_root / "tool_metadata" / "tool_description_candidates.json"
    tool_description_candidates = read_json(tool_description_candidate_path, [])
    if not isinstance(tool_description_candidates, list):
        tool_description_candidates = []

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    training_library_payload = {
        "library_version": 3,
        "aggregation_mode": "narrative",
        "dataset_name": training_root.name,
        "generated_at": generated_at,
        "raw_template_count": len(raw_templates),
        "cluster_count": cluster_payload.get("cluster_count", 0),
        "clusters": cluster_payload.get("clusters", []),
        "pattern_count": len(runtime_templates),
        "patterns": runtime_templates,
        "templates": runtime_templates,
    }
    runtime_library_payload = {
        "library_version": 3,
        "aggregation_mode": "narrative",
        "dataset_name": training_root.name,
        "generated_at": generated_at,
        "pattern_count": len(runtime_templates),
        "patterns": runtime_templates,
        "template_count": len(runtime_templates),
        "templates": runtime_templates,
    }

    question_summary_path = output_root / "distilled" / "question_training_summaries.json"
    raw_template_path = output_root / "distilled" / "raw_templates.json"
    failure_summary_path = output_root / "failures" / "failed_question_summaries.json"
    cluster_path = output_root / "clusters" / "template_clusters.json"
    merge_decision_path = output_root / "clusters" / "merge_decisions.jsonl"
    library_output_path = output_root / "library" / "strategy_library.json"
    template_source_index_path = output_root / "library" / "template_source_index.json"
    report_path = output_root / "report.md"
    manifest_path = output_root / "manifests" / "recluster_manifest.json"

    manifest = {
        "training_root": str(training_root),
        "question_detail_dir": str(detail_dir),
        "question_count": len(detail_paths),
        "successful_question_count": len(question_summaries),
        "failed_question_count": len(failed_summaries),
        "runtime_library_path": str(runtime_library_path),
        "runtime_tool_metadata_dir": str(runtime_tool_metadata_dir),
        "generated_at": generated_at,
        "mode": "recluster_from_existing_per_question_artifacts",
    }

    json_dump_atomic(str(manifest_path), manifest)
    json_dump_atomic(str(question_summary_path), question_summaries)
    json_dump_atomic(str(raw_template_path), raw_templates)
    json_dump_atomic(str(failure_summary_path), failed_summaries)
    json_dump_atomic(str(cluster_path), cluster_payload)
    write_jsonl(merge_decision_path, cluster_manager.merge_decisions)
    json_dump_atomic(str(library_output_path), training_library_payload)
    json_dump_atomic(str(template_source_index_path), template_source_index)
    json_dump_atomic(str(runtime_library_path), runtime_library_payload)
    runtime_tool_metadata_paths = runtime_asset_manager.export_tool_metadata_overrides(tool_description_candidates)
    runtime_template_source_index_path = runtime_asset_manager.export_template_source_index(template_source_index)

    q43_cluster_id = ""
    q43_pattern_name = ""
    for item in question_summaries:
        if str(item.get("question_id", "")) == "q43":
            q43_cluster_id = str(item.get("cluster_id", "") or "")
            break
    for cluster in cluster_payload.get("clusters", []):
        if str(cluster.get("cluster_id", "") or "") == q43_cluster_id:
            q43_pattern_name = str(cluster.get("pattern_name", "") or "")
            break

    report_lines = [
        "# Strategy Recluster Report",
        "",
        f"- Training root: `{training_root}`",
        f"- Successful questions: `{len(question_summaries)}`",
        f"- Failed questions: `{len(failed_summaries)}`",
        f"- Raw template count: `{len(raw_templates)}`",
        f"- Cluster count: `{cluster_payload.get('cluster_count', 0)}`",
        f"- Consolidation merges: `{consolidation_merge_count}`",
        f"- Runtime library: `{runtime_library_path}`",
        f"- q43 cluster: `{q43_cluster_id}`",
        f"- q43 pattern: `{q43_pattern_name}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "question_summary_path": str(question_summary_path),
                "raw_template_path": str(raw_template_path),
                "failure_summary_path": str(failure_summary_path),
                "cluster_path": str(cluster_path),
                "merge_decision_path": str(merge_decision_path),
                "library_output_path": str(library_output_path),
                "template_source_index_path": str(template_source_index_path),
                "runtime_library_path": str(runtime_library_path),
                "runtime_tool_metadata_paths": runtime_tool_metadata_paths,
                "runtime_template_source_index_path": runtime_template_source_index_path,
                "report_path": str(report_path),
                "successful_question_count": len(question_summaries),
                "failed_question_count": len(failed_summaries),
                "cluster_count": cluster_payload.get("cluster_count", 0),
                "consolidation_merge_count": consolidation_merge_count,
                "q43_cluster_id": q43_cluster_id,
                "q43_pattern_name": q43_pattern_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
