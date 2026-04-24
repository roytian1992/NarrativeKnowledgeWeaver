from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.knowledge_extraction_agent import InformationExtractionAgent, clean_screenplay_text
from core.functions.regular_functions.entity_schema_grounding import EntitySchemaGrounder
from core.functions.regular_functions.event_occasion_frame_extraction import EventOccasionFrameExtractor
from core.functions.regular_functions.open_entity_extraction import OpenEntityExtractor
from core.model_providers.openai_llm import OpenAILLM
from core.utils.config import load_config
from core.utils.prompt_loader import YAMLPromptLoader
from scripts.compare_event_first_pipeline import (
    _auto_relations_from_frames,
    _collect_frame_entity_candidates,
    _dedup_grounded_entities,
    _frame_nodes,
    _known_entities_text,
)
from scripts.compare_relation_modes_on_workspace import _dump_json, _load_json


def _now() -> float:
    return time.perf_counter()


def _round(value: float) -> float:
    return round(float(value), 3)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean_sec": 0.0, "min_sec": 0.0, "max_sec": 0.0, "stdev_sec": 0.0}
    result = {
        "mean_sec": _round(statistics.mean(values)),
        "min_sec": _round(min(values)),
        "max_sec": _round(max(values)),
        "stdev_sec": _round(statistics.stdev(values)) if len(values) >= 2 else 0.0,
    }
    return result


def _build_agent(config: Any) -> InformationExtractionAgent:
    entity_schema = _load_json(REPO_ROOT / config.global_config.schema_dir / "default_entity_schema.json")
    relation_schema = _load_json(REPO_ROOT / config.global_config.schema_dir / "default_relation_schema.json")
    llm = OpenAILLM(
        config,
        timeout=int(getattr(config.llm, "timeout", 180) or 180),
        max_retries=0,
    )
    return InformationExtractionAgent(
        config=config,
        llm=llm,
        entity_schema=entity_schema,
        relation_schema=relation_schema,
        memory_store=None,
    )


def _build_event_first_helpers(config: Any, llm: Any) -> Dict[str, Any]:
    prompt_loader = YAMLPromptLoader(config.global_config.prompt_dir)
    return {
        "frame_extractor": EventOccasionFrameExtractor(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "event_occasion_frame_extraction_task.json"),
        ),
        "open_entity_extractor": OpenEntityExtractor(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "open_entity_extraction_task.json"),
        ),
        "entity_grounder": EntitySchemaGrounder(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "entity_schema_grounding_task.json"),
            entity_schema_path=str(REPO_ROOT / config.global_config.schema_dir / "default_entity_schema.json"),
        ),
    }


def _run_schema_direct_full(
    *,
    agent: InformationExtractionAgent,
    cleaned_text: str,
    rid_namespace: str,
) -> Dict[str, Any]:
    agent.config.knowledge_graph_builder.relation_extraction_mode = "schema_direct"

    t0 = _now()
    entities = agent._extract_entities_one_chunk(
        cleaned_text=cleaned_text,
        prev_entities=[],
        memory_context="",
    )
    t1 = _now()

    raw_relations_map, feedbacks = agent._extract_relations_one_chunk(
        cleaned_text=cleaned_text,
        entities=copy.deepcopy(entities),
        prev_all_relations={},
        rid_namespace=rid_namespace,
        memory_context="",
    )
    t2 = _now()

    entities_after_fix, fixed_relations_map = agent._resolve_errors(
        entities=copy.deepcopy(entities),
        all_relations=copy.deepcopy(raw_relations_map),
        all_feedbacks=copy.deepcopy(feedbacks),
        content=cleaned_text,
    )
    fixed_relations_map = agent._repair_relation_coverage(
        cleaned_text=cleaned_text,
        entities=entities_after_fix,
        all_relations=fixed_relations_map,
        rid_namespace=rid_namespace,
        memory_context="",
    )
    t3 = _now()

    final_relations_map = agent._dedup_multi_relations(
        all_relations=copy.deepcopy(fixed_relations_map),
        content=cleaned_text,
    )
    t4 = _now()

    return {
        "counts": {
            "entity_count": len(entities_after_fix),
            "raw_relation_count": len(raw_relations_map),
            "fixed_relation_count": len(fixed_relations_map),
            "final_relation_count": len(final_relations_map),
            "feedback_count": sum(len(v or []) for v in feedbacks.values()),
        },
        "timings_sec": {
            "entity_extraction": _round(t1 - t0),
            "relation_extraction": _round(t2 - t1),
            "fix_and_coverage": _round(t3 - t2),
            "dedup": _round(t4 - t3),
            "total": _round(t4 - t0),
        },
    }


def _run_event_first_full(
    *,
    agent: InformationExtractionAgent,
    helpers: Dict[str, Any],
    cleaned_text: str,
    rid_namespace: str,
) -> Dict[str, Any]:
    frame_extractor = helpers["frame_extractor"]
    open_entity_extractor = helpers["open_entity_extractor"]
    entity_grounder = helpers["entity_grounder"]

    t0 = _now()
    frames = json.loads(frame_extractor.call(json.dumps({"text": cleaned_text, "memory_context": ""})))
    t1 = _now()

    event_nodes, occasion_nodes = _frame_nodes(frames)
    frame_node_names = {str(item["name"]).strip().lower() for item in event_nodes + occasion_nodes}
    attached_candidates = _collect_frame_entity_candidates(frames, frame_node_names=frame_node_names)

    open_entities = json.loads(
        open_entity_extractor.call(
            json.dumps(
                {
                    "text": cleaned_text,
                    "known_entities": _known_entities_text(event_nodes, occasion_nodes, attached_candidates),
                    "memory_context": "",
                }
            )
        )
    )
    t2 = _now()

    grounded_attached_entities = json.loads(
        entity_grounder.call(
            json.dumps(
                {
                    "text": cleaned_text,
                    "open_entities": attached_candidates,
                    "memory_context": "",
                }
            )
        )
    )
    t3 = _now()

    grounded_open_entities = json.loads(
        entity_grounder.call(
            json.dumps(
                {
                    "text": cleaned_text,
                    "open_entities": open_entities,
                    "memory_context": "",
                }
            )
        )
    )
    t4 = _now()

    grounded_entities = _dedup_grounded_entities(grounded_attached_entities + grounded_open_entities)
    auto_relations = _auto_relations_from_frames(
        frames=frames,
        grounded_entities=grounded_attached_entities,
        event_nodes=event_nodes,
        occasion_nodes=occasion_nodes,
    )
    t5 = _now()

    agent.config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"
    open_rel_map, open_feedbacks = agent._extract_relations_one_chunk(
        cleaned_text=cleaned_text,
        entities=copy.deepcopy(grounded_open_entities),
        prev_all_relations={},
        rid_namespace=rid_namespace,
        memory_context="",
    )
    _, fixed_open_rel_map = agent._resolve_errors(
        entities=copy.deepcopy(grounded_open_entities),
        all_relations=copy.deepcopy(open_rel_map),
        all_feedbacks=copy.deepcopy(open_feedbacks),
        content=cleaned_text,
    )
    t6 = _now()

    combined_rel_map = dict(auto_relations)
    combined_rel_map.update(fixed_open_rel_map)
    final_relations_map = agent._dedup_multi_relations(
        all_relations=combined_rel_map,
        content=cleaned_text,
    )
    t7 = _now()

    return {
        "counts": {
            "event_count": len(event_nodes),
            "occasion_count": len(occasion_nodes),
            "attached_entity_candidate_count": len(attached_candidates),
            "open_entity_count": len(open_entities),
            "grounded_attached_entity_count": len(grounded_attached_entities),
            "grounded_open_entity_count": len(grounded_open_entities),
            "grounded_entity_count": len(grounded_entities),
            "auto_relation_count": len(auto_relations),
            "open_relation_count_after_fix": len(fixed_open_rel_map),
            "final_relation_count": len(final_relations_map),
            "feedback_count": sum(len(v or []) for v in open_feedbacks.values()),
        },
        "timings_sec": {
            "frame_extraction": _round(t1 - t0),
            "open_entity_extraction": _round(t2 - t1),
            "attached_entity_grounding": _round(t3 - t2),
            "open_entity_grounding": _round(t4 - t3),
            "auto_relation_build": _round(t5 - t4),
            "open_relation_extract_and_fix": _round(t6 - t5),
            "dedup": _round(t7 - t6),
            "total": _round(t7 - t0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full schema_direct pipeline vs event-first pipeline on one chunk.")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model-name", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--aggressive-clean", action="store_true", default=True)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    doc2chunks = _load_json(workspace_dir / "knowledge_graph" / "doc2chunks.json")
    chunk = doc2chunks[args.document_id]["chunks"][args.chunk_index]
    raw_text = str(chunk.get("content", "")).strip()
    cleaned_text = clean_screenplay_text(raw_text, aggressive=bool(args.aggressive_clean))

    config = load_config(str(Path(args.config).resolve()))
    config.llm.timeout = int(args.llm_timeout)
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    if args.llm_model_name:
        config.llm.model_name = args.llm_model_name

    agent = _build_agent(config)
    helpers = _build_event_first_helpers(config, agent.llm)

    warmups: List[Dict[str, Any]] = []
    for i in range(max(0, int(args.warmup_runs))):
        schema_warm = _run_schema_direct_full(
            agent=agent,
            cleaned_text=cleaned_text,
            rid_namespace=f"warmup.schema_direct.{i}",
        )
        event_warm = _run_event_first_full(
            agent=agent,
            helpers=helpers,
            cleaned_text=cleaned_text,
            rid_namespace=f"warmup.event_first.{i}",
        )
        warmups.append(
            {
                "schema_direct_total_sec": schema_warm["timings_sec"]["total"],
                "event_first_total_sec": event_warm["timings_sec"]["total"],
            }
        )

    measured: List[Dict[str, Any]] = []
    schema_totals: List[float] = []
    event_totals: List[float] = []

    for i in range(max(1, int(args.repeats))):
        schema_run = _run_schema_direct_full(
            agent=agent,
            cleaned_text=cleaned_text,
            rid_namespace=f"measure.schema_direct.{i}",
        )
        event_run = _run_event_first_full(
            agent=agent,
            helpers=helpers,
            cleaned_text=cleaned_text,
            rid_namespace=f"measure.event_first.{i}",
        )
        schema_totals.append(float(schema_run["timings_sec"]["total"]))
        event_totals.append(float(event_run["timings_sec"]["total"]))
        measured.append(
            {
                "repeat_index": i,
                "schema_direct": schema_run,
                "event_first": event_run,
            }
        )

    schema_avg = statistics.mean(schema_totals)
    event_avg = statistics.mean(event_totals)
    abs_saved = schema_avg - event_avg
    pct_saved = (abs_saved / schema_avg * 100.0) if schema_avg > 0 else 0.0

    summary = {
        "workspace_dir": str(workspace_dir),
        "document_id": args.document_id,
        "chunk_index": args.chunk_index,
        "chunk_id": chunk.get("id", ""),
        "text_word_count": len(cleaned_text.split()),
        "llm_model_name": config.llm.model_name,
        "llm_base_url": config.llm.base_url,
        "warmup_runs": int(args.warmup_runs),
        "repeats": int(args.repeats),
        "warmup_totals": warmups,
        "schema_direct_total_sec": _stats(schema_totals),
        "event_first_total_sec": _stats(event_totals),
        "avg_time_saved_sec": _round(abs_saved),
        "avg_time_saved_pct": _round(pct_saved),
        "measured_runs": measured,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(output_dir / "benchmark_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
