from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.functions.regular_functions.entity_schema_grounding import EntitySchemaGrounder
from core.functions.regular_functions.event_occasion_frame_extraction import EventOccasionFrameExtractor
from core.functions.regular_functions.open_entity_extraction import OpenEntityExtractor
from core.model_providers.openai_llm import OpenAILLM
from core.utils.config import load_config
from core.utils.prompt_loader import YAMLPromptLoader
from scripts.compare_relation_modes_on_workspace import (
    _compare_sets,
    _dump_json,
    _load_json,
    _normalize_rel,
    _relation_type_distribution,
)


def _norm_name(text: Any) -> str:
    return str(text or "").strip().lower()


def _dedup_open_entities(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not name:
            continue
        key = name.lower()
        cur = merged.get(key)
        if cur is None or len(desc) > len(str(cur.get("description", ""))):
            merged[key] = {"name": name, "description": desc}
    return list(merged.values())


def _is_low_value_related_entity(name: str, description: str) -> bool:
    key = _norm_name(name)
    desc = _norm_name(description)
    weak_markers = [
        "消息",
        "news of",
        "report of",
        "rumor",
        "传闻",
        "流言",
    ]
    return any(marker in key or marker in desc for marker in weak_markers)


def _is_communicative_event(name: str, description: str) -> bool:
    text = f"{_norm_name(name)} {_norm_name(description)}"
    markers = [
        "tell",
        "telling",
        "inform",
        "discuss",
        "discussion",
        "talk",
        "talking",
        "read",
        "reading",
        "mention",
        "listing",
        "list",
        "conversation",
        "dialogue",
    ]
    return any(marker in text for marker in markers)


def _collect_frame_entity_candidates(
    frames: Dict[str, Any],
    *,
    frame_node_names: set[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    fields = [
        "agents",
        "experiencers",
        "patients",
        "participants",
        "initiators",
        "locations",
        "times",
        "related_entities",
    ]
    for group_key in ["events", "occasions"]:
        for frame in frames.get(group_key, []):
            for field in fields:
                for item in frame.get(field, []):
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    name = str(item.get("name", "")).strip()
                    description = str(item.get("description", "")).strip()
                    if _norm_name(name) in frame_node_names:
                        continue
                    if field == "related_entities" and _is_low_value_related_entity(name, description):
                        continue
                    out.append({"name": name, "description": description})
    return _dedup_open_entities(out)


def _frame_nodes(frames: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events = []
    for item in frames.get("events", []):
        events.append({"type": "Event", "name": item["name"], "description": item["description"], "scope": "local"})
    occasions = []
    for item in frames.get("occasions", []):
        occasions.append({"type": "Occasion", "name": item["name"], "description": item["description"], "scope": "global"})
    return events, occasions


def _known_entities_text(event_nodes: List[Dict[str, Any]], occasion_nodes: List[Dict[str, Any]], attached_candidates: List[Dict[str, str]]) -> str:
    lines = []
    for item in event_nodes + occasion_nodes:
        lines.append(item["name"])
    for item in attached_candidates:
        lines.append(item["name"])
    return "\n".join(sorted(set(x for x in lines if x)))


def _dedup_grounded_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name or not etype:
            continue
        key = (name.lower(), etype)
        cur = merged.get(key)
        if cur is None or len(str(item.get("description", ""))) > len(str(cur.get("description", ""))):
            merged[key] = item
    return list(merged.values())


def _entity_index(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(e.get("name", "")).strip(): e for e in entities if isinstance(e, dict) and str(e.get("name", "")).strip()}


def _make_relation(rid: str, subject: str, object_: str, relation_type: str, relation_name: str, description: str) -> Dict[str, Any]:
    return {
        "rid": rid,
        "subject": subject,
        "object": object_,
        "relation_type": relation_type,
        "relation_name": relation_name,
        "description": description,
        "conf": 0.8,
    }


def _auto_relations_from_frames(
    *,
    frames: Dict[str, Any],
    grounded_entities: List[Dict[str, Any]],
    event_nodes: List[Dict[str, Any]],
    occasion_nodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    idx = _entity_index(grounded_entities + event_nodes + occasion_nodes)
    rid_i = 1
    out: Dict[str, Dict[str, Any]] = {}
    seen_triples = set()

    def add_if_valid(subject: str, object_: str, relation_type: str, relation_name: str, description: str) -> None:
        nonlocal rid_i
        triple = (subject, relation_type, object_)
        if subject not in idx or object_ not in idx or subject == object_ or triple in seen_triples:
            return
        seen_triples.add(triple)
        key = f"auto#{rid_i}"
        rid_i += 1
        out[key] = _make_relation(key, subject, object_, relation_type, relation_name, description)

    for event in frames.get("events", []):
        ev_name = event["name"]
        is_communicative = _is_communicative_event(ev_name, str(event.get("description", "")))
        event_locations = []
        for item in event.get("agents", []):
            add_if_valid(item["name"], ev_name, "performs", "performs", f"{item['name']} performs event [{ev_name}].")
        for item in event.get("experiencers", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Character":
                add_if_valid(item["name"], ev_name, "experiences", "experiences", f"{item['name']} experiences event [{ev_name}].")
        if not is_communicative:
            for item in event.get("patients", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") in {"Character", "Object", "Concept"}:
                    add_if_valid(item["name"], ev_name, "undergoes", "undergoes", f"{item['name']} undergoes event [{ev_name}].")
        for item in event.get("locations", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Location":
                event_locations.append(item["name"])
                add_if_valid(ev_name, item["name"], "occurs_at", "occurs at", f"Event [{ev_name}] occurs at [{item['name']}].")
        for item in event.get("times", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "TimePoint":
                add_if_valid(ev_name, item["name"], "occurs_on", "occurs on", f"Event [{ev_name}] occurs on [{item['name']}].")
        for item in event.get("occasion", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Occasion":
                add_if_valid(ev_name, item["name"], "occurs_during", "occurs during", f"Event [{ev_name}] occurs during [{item['name']}].")
        if len(event_locations) == 1:
            event_location = event_locations[0]
            for field in ["agents", "experiencers"]:
                for item in event.get(field, []):
                    ent = idx.get(item["name"], {})
                    if ent.get("type") in {"Character", "Object"}:
                        add_if_valid(item["name"], event_location, "located_at", "located at", f"[{item['name']}] is located at [{event_location}] during the event [{ev_name}].")
            for item in event.get("related_entities", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") == "Location":
                    add_if_valid(item["name"], event_location, "part_of", "part of", f"[{item['name']}] is a sub-location within [{event_location}].")
                elif ent.get("type") == "Object":
                    add_if_valid(item["name"], event_location, "located_at", "located at", f"[{item['name']}] is located at [{event_location}].")

    for occasion in frames.get("occasions", []):
        oc_name = occasion["name"]
        occasion_locations = []
        for item in occasion.get("participants", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") in {"Character", "Concept"}:
                add_if_valid(item["name"], oc_name, "participates_in", "participates in", f"{item['name']} participates in occasion [{oc_name}].")
        for item in occasion.get("initiators", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") in {"Character", "Concept"}:
                add_if_valid(item["name"], oc_name, "initiates", "initiates", f"{item['name']} initiates occasion [{oc_name}].")
        for item in occasion.get("locations", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Location":
                occasion_locations.append(item["name"])
                add_if_valid(oc_name, item["name"], "occurs_at", "occurs at", f"Occasion [{oc_name}] occurs at [{item['name']}].")
        for item in occasion.get("times", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "TimePoint":
                add_if_valid(oc_name, item["name"], "occurs_on", "occurs on", f"Occasion [{oc_name}] occurs on [{item['name']}].")
        if len(occasion_locations) == 1:
            occasion_location = occasion_locations[0]
            add_if_valid(occasion_location, oc_name, "part_of", "part of", f"[{occasion_location}] is a location that forms part of occasion [{oc_name}].")
            for item in occasion.get("participants", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") in {"Character", "Object"}:
                    add_if_valid(item["name"], occasion_location, "located_at", "located at", f"[{item['name']}] is located at [{occasion_location}] during occasion [{oc_name}].")
            for item in occasion.get("related_entities", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") == "Location":
                    add_if_valid(item["name"], occasion_location, "part_of", "part of", f"[{item['name']}] is a sub-location within [{occasion_location}].")
                elif ent.get("type") == "Object":
                    add_if_valid(item["name"], occasion_location, "located_at", "located at", f"[{item['name']}] is located at [{occasion_location}] during occasion [{oc_name}].")

    return out


def _summary_entities(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ent in entities:
        t = str(ent.get("type", "")).strip()
        if not t:
            continue
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experimental event-first pipeline against schema_direct baseline.")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model-name", default="")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    config = load_config(str(Path(args.config).resolve()))
    config.llm.timeout = int(args.llm_timeout)
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    if args.llm_model_name:
        config.llm.model_name = args.llm_model_name

    llm = OpenAILLM(config, timeout=int(getattr(config.llm, "timeout", 180) or 180), max_retries=0)
    prompt_loader = YAMLPromptLoader(config.global_config.prompt_dir)

    frame_extractor = EventOccasionFrameExtractor(
        prompt_loader=prompt_loader,
        llm=llm,
        task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "event_occasion_frame_extraction_task.json"),
    )
    open_entity_extractor = OpenEntityExtractor(
        prompt_loader=prompt_loader,
        llm=llm,
        task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "open_entity_extraction_task.json"),
    )
    entity_grounder = EntitySchemaGrounder(
        prompt_loader=prompt_loader,
        llm=llm,
        task_schema_path=str(REPO_ROOT / config.global_config.task_dir / "entity_schema_grounding_task.json"),
        entity_schema_path=str(REPO_ROOT / config.global_config.schema_dir / "default_entity_schema.json"),
    )

    entity_schema = _load_json(REPO_ROOT / config.global_config.schema_dir / "default_entity_schema.json")
    relation_schema = _load_json(REPO_ROOT / config.global_config.schema_dir / "default_relation_schema.json")
    agent = InformationExtractionAgent(config=config, llm=llm, entity_schema=entity_schema, relation_schema=relation_schema, memory_store=None)

    doc2chunks = _load_json(workspace_dir / "knowledge_graph" / "doc2chunks.json")
    chunk = doc2chunks[args.document_id]["chunks"][args.chunk_index]
    text = str(chunk.get("content", "")).strip()

    baseline = None
    try:
        from scripts.compare_relation_modes_on_workspace import _run_mode
        baseline = _run_mode(
            agent=agent,
            mode="schema_direct",
            text=text,
            entities=copy.deepcopy(_load_json(workspace_dir / "knowledge_graph" / "extraction_results.json")[args.document_id]["entities"]),
            rid_namespace=f"{args.document_id}:baseline",
        )
    except Exception:
        baseline = None

    frames = json.loads(frame_extractor.call(json.dumps({"text": text, "memory_context": ""})))
    event_nodes, occasion_nodes = _frame_nodes(frames)
    frame_node_names = {_norm_name(item["name"]) for item in event_nodes + occasion_nodes}
    attached_candidates = _collect_frame_entity_candidates(frames, frame_node_names=frame_node_names)

    open_entities = json.loads(
        open_entity_extractor.call(
            json.dumps(
                {
                    "text": text,
                    "known_entities": _known_entities_text(event_nodes, occasion_nodes, attached_candidates),
                    "memory_context": "",
                }
            )
        )
    )
    grounded_attached_entities = json.loads(
        entity_grounder.call(
            json.dumps(
                {
                    "text": text,
                    "open_entities": attached_candidates,
                    "memory_context": "",
                }
            )
        )
    )
    grounded_open_entities = json.loads(
        entity_grounder.call(
            json.dumps(
                {
                    "text": text,
                    "open_entities": open_entities,
                    "memory_context": "",
                }
            )
        )
    )
    grounded_entities = _dedup_grounded_entities(grounded_attached_entities + grounded_open_entities)

    auto_relations = _auto_relations_from_frames(
        frames=frames,
        grounded_entities=grounded_attached_entities,
        event_nodes=event_nodes,
        occasion_nodes=occasion_nodes,
    )

    non_induced_entities = grounded_open_entities
    agent.config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"
    open_rel_map, open_feedbacks = agent._extract_relations_one_chunk(
        cleaned_text=text,
        entities=non_induced_entities,
        prev_all_relations={},
        rid_namespace=f"{args.document_id}:event_first.open_rel",
        memory_context="",
    )
    _, fixed_open_rel_map = agent._resolve_errors(
        entities=copy.deepcopy(non_induced_entities),
        all_relations=copy.deepcopy(open_rel_map),
        all_feedbacks=copy.deepcopy(open_feedbacks),
        content=text,
    )

    combined_rel_map = dict(auto_relations)
    for rid, rel in fixed_open_rel_map.items():
        combined_rel_map[rid] = rel
    combined_rel_map = agent._dedup_multi_relations(all_relations=combined_rel_map, content=text)

    experimental_entities = event_nodes + occasion_nodes + grounded_entities
    experimental_final_relations = [_normalize_rel(rel) for _, rel in sorted(combined_rel_map.items())]

    summary = {
        "document_id": args.document_id,
        "chunk_index": args.chunk_index,
        "chunk_id": chunk.get("id", ""),
        "text_word_count": len(text.split()),
        "experimental": {
            "event_count": len(event_nodes),
            "occasion_count": len(occasion_nodes),
            "attached_entity_candidate_count": len(attached_candidates),
            "open_entity_count": len(open_entities),
            "grounded_attached_entity_count": len(grounded_attached_entities),
            "grounded_open_entity_count": len(grounded_open_entities),
            "grounded_entity_count": len(grounded_entities),
            "entity_type_distribution": _summary_entities(experimental_entities),
            "auto_relation_count": len(auto_relations),
            "open_relation_count_after_fix": len(fixed_open_rel_map),
            "final_relation_count": len(experimental_final_relations),
            "final_relation_type_distribution": _relation_type_distribution(experimental_final_relations),
        },
    }
    if baseline is not None:
        baseline_final = baseline["final_relations_after_dedup"]
        summary["baseline_schema_direct"] = {
            "final_relation_count": len(baseline_final),
            "final_relation_type_distribution": _relation_type_distribution(baseline_final),
        }
        summary["final_compare_vs_baseline"] = _compare_sets(baseline_final, experimental_final_relations)

    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(output_dir / "frames.json", frames)
    _dump_json(output_dir / "event_nodes.json", event_nodes)
    _dump_json(output_dir / "occasion_nodes.json", occasion_nodes)
    _dump_json(output_dir / "attached_entity_candidates.json", attached_candidates)
    _dump_json(output_dir / "open_entities.json", open_entities)
    _dump_json(output_dir / "grounded_attached_entities.json", grounded_attached_entities)
    _dump_json(output_dir / "grounded_open_entities.json", grounded_open_entities)
    _dump_json(output_dir / "grounded_entities.json", grounded_entities)
    _dump_json(output_dir / "auto_relations.json", [_normalize_rel(v) for _, v in sorted(auto_relations.items())])
    _dump_json(output_dir / "open_relations_after_fix.json", [_normalize_rel(v) for _, v in sorted(fixed_open_rel_map.items())])
    _dump_json(output_dir / "experimental_relations_final.json", experimental_final_relations)
    if baseline is not None:
        _dump_json(output_dir / "baseline_schema_direct_final.json", baseline["final_relations_after_dedup"])
    _dump_json(output_dir / "comparison_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
