from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.knowledge_extraction_agent import InformationExtractionAgent, clean_screenplay_text
from core.model_providers.openai_llm import OpenAILLM
from core.utils.config import load_config


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_rel(rel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rid": rel.get("rid", ""),
        "subject": rel.get("subject", ""),
        "object": rel.get("object", ""),
        "relation_type": rel.get("relation_type", rel.get("type", "")),
        "relation_name": rel.get("relation_name", ""),
        "description": rel.get("description", ""),
        "conf": rel.get("conf"),
        "properties": rel.get("properties", {}),
        "persistence": rel.get("persistence"),
        "auto_fixed": rel.get("auto_fixed", False),
        "fix_reason": rel.get("fix_reason", ""),
    }


def _feedback_bucket_counts(feedbacks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    return {k: len(v or []) for k, v in sorted((feedbacks or {}).items())}


def _relation_type_distribution(relations: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for rel in relations or []:
        rtype = str(rel.get("relation_type", rel.get("type", "")) or "").strip()
        if rtype:
            counter[rtype] += 1
    return dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def _triple_key(rel: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(rel.get("subject", "")).strip(),
        str(rel.get("relation_type", rel.get("type", ""))).strip(),
        str(rel.get("object", "")).strip(),
    )


def _pair_type_distribution(relations: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> Dict[str, int]:
    name2type = {
        str(ent.get("name", "")).strip(): str(ent.get("type", "")).strip()
        for ent in entities or []
        if isinstance(ent, dict) and ent.get("name") and ent.get("type")
    }
    counter = Counter()
    for rel in relations or []:
        subj = str(rel.get("subject", "")).strip()
        obj = str(rel.get("object", "")).strip()
        st = name2type.get(subj, "UNKNOWN")
        ot = name2type.get(obj, "UNKNOWN")
        rtype = str(rel.get("relation_type", rel.get("type", ""))).strip() or "UNKNOWN"
        counter[f"{st} -> {rtype} -> {ot}"] += 1
    return dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def _summarize_mode(
    *,
    mode: str,
    raw_relations: List[Dict[str, Any]],
    feedbacks_before_fix: Dict[str, List[Dict[str, Any]]],
    fixed_relations: List[Dict[str, Any]],
    final_relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "counts": {
            "raw_relations": len(raw_relations),
            "fixed_relations_before_dedup": len(fixed_relations),
            "final_relations_after_dedup": len(final_relations),
            "feedback_total_before_fix": sum(len(v or []) for v in (feedbacks_before_fix or {}).values()),
        },
        "feedback_bucket_counts_before_fix": _feedback_bucket_counts(feedbacks_before_fix),
        "relation_type_distribution_raw": _relation_type_distribution(raw_relations),
        "relation_type_distribution_final": _relation_type_distribution(final_relations),
        "subject_object_type_pattern_distribution_final": _pair_type_distribution(final_relations, entities),
    }


def _compare_sets(old_relations: List[Dict[str, Any]], new_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    old_map = {_triple_key(rel): rel for rel in old_relations or []}
    new_map = {_triple_key(rel): rel for rel in new_relations or []}
    old_keys = set(old_map)
    new_keys = set(new_map)
    overlap = sorted(old_keys & new_keys)
    old_only = sorted(old_keys - new_keys)
    new_only = sorted(new_keys - old_keys)
    return {
        "overlap_count": len(overlap),
        "old_only_count": len(old_only),
        "new_only_count": len(new_only),
        "overlap_triples": [list(x) for x in overlap],
        "old_only_triples": [list(x) for x in old_only],
        "new_only_triples": [list(x) for x in new_only],
    }


def _render_markdown_report(
    *,
    workspace_dir: Path,
    document_id: str,
    chunk_index: int,
    chunk_id: str,
    text_word_count: int,
    entity_count: int,
    old_summary: Dict[str, Any],
    new_summary: Dict[str, Any],
    final_compare: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Relation Mode Comparison")
    lines.append("")
    lines.append(f"- workspace: `{workspace_dir}`")
    lines.append(f"- document_id: `{document_id}`")
    lines.append(f"- chunk_index: `{chunk_index}`")
    lines.append(f"- chunk_id: `{chunk_id}`")
    lines.append(f"- text_word_count: `{text_word_count}`")
    lines.append(f"- frozen_entity_count: `{entity_count}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| mode | raw | fixed_before_dedup | final_after_dedup | feedback_before_fix |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for summary in [old_summary, new_summary]:
        counts = summary["counts"]
        lines.append(
            f"| `{summary['mode']}` | {counts['raw_relations']} | {counts['fixed_relations_before_dedup']} | "
            f"{counts['final_relations_after_dedup']} | {counts['feedback_total_before_fix']} |"
        )
    lines.append("")
    lines.append("## Final Overlap")
    lines.append("")
    lines.append(f"- overlap_count: {final_compare['overlap_count']}")
    lines.append(f"- old_only_count: {final_compare['old_only_count']}")
    lines.append(f"- new_only_count: {final_compare['new_only_count']}")
    lines.append("")
    lines.append("## Feedback Buckets Before Fix")
    lines.append("")
    for summary in [old_summary, new_summary]:
        lines.append(f"### `{summary['mode']}`")
        bucket_counts = summary["feedback_bucket_counts_before_fix"]
        if not bucket_counts:
            lines.append("- none")
        else:
            for key, value in bucket_counts.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Final Relation Type Distribution")
    lines.append("")
    for summary in [old_summary, new_summary]:
        lines.append(f"### `{summary['mode']}`")
        dist = summary["relation_type_distribution_final"]
        if not dist:
            lines.append("- none")
        else:
            for key, value in dist.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    if final_compare["new_only_triples"]:
        lines.append("## New-Only Final Triples")
        lines.append("")
        for subj, rtype, obj in final_compare["new_only_triples"]:
            lines.append(f"- ({subj}) -[{rtype}]-> ({obj})")
        lines.append("")
    if final_compare["old_only_triples"]:
        lines.append("## Old-Only Final Triples")
        lines.append("")
        for subj, rtype, obj in final_compare["old_only_triples"]:
            lines.append(f"- ({subj}) -[{rtype}]-> ({obj})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _run_mode(
    *,
    agent: InformationExtractionAgent,
    mode: str,
    text: str,
    entities: List[Dict[str, Any]],
    rid_namespace: str,
) -> Dict[str, Any]:
    print(f"[compare] start mode={mode}", flush=True)
    agent.config.knowledge_graph_builder.relation_extraction_mode = mode
    frozen_entities = copy.deepcopy(entities)
    print(f"[compare] extracting raw relations mode={mode}", flush=True)
    raw_relations_map, feedbacks = agent._extract_relations_one_chunk(
        cleaned_text=text,
        entities=frozen_entities,
        prev_all_relations={},
        rid_namespace=rid_namespace,
        memory_context="",
    )
    raw_relations = [_normalize_rel(rel) for _, rel in sorted(raw_relations_map.items())]
    print(
        f"[compare] extracted raw relations mode={mode} count={len(raw_relations)} feedbacks={sum(len(v or []) for v in feedbacks.values())}",
        flush=True,
    )

    print(f"[compare] resolving errors mode={mode}", flush=True)
    entities_after_fix, fixed_relations_map = agent._resolve_errors(
        entities=copy.deepcopy(entities),
        all_relations=copy.deepcopy(raw_relations_map),
        all_feedbacks=copy.deepcopy(feedbacks),
        content=text,
    )
    fixed_relations_map = agent._repair_relation_coverage(
        cleaned_text=text,
        entities=entities_after_fix,
        all_relations=fixed_relations_map,
        rid_namespace=rid_namespace,
        memory_context="",
    )
    fixed_relations = [_normalize_rel(rel) for _, rel in sorted(fixed_relations_map.items())]
    print(f"[compare] resolved errors mode={mode} relations={len(fixed_relations)}", flush=True)

    print(f"[compare] deduping relations mode={mode}", flush=True)
    final_relations_map = agent._dedup_multi_relations(
        all_relations=copy.deepcopy(fixed_relations_map),
        content=text,
    )
    final_relations = [_normalize_rel(rel) for _, rel in sorted(final_relations_map.items())]
    print(f"[compare] finished mode={mode} final_relations={len(final_relations)}", flush=True)

    return {
        "mode": mode,
        "entities_after_fix_count": len(entities_after_fix),
        "raw_relations": raw_relations,
        "feedbacks_before_fix": feedbacks,
        "fixed_relations_before_dedup": fixed_relations,
        "final_relations_after_dedup": final_relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare schema_direct vs open_then_ground on one workspace chunk.")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--old-mode", default="schema_direct")
    parser.add_argument("--new-mode", default="open_then_ground")
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model-name", default="")
    parser.add_argument("--aggressive-clean", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    os.chdir(repo_root)

    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    document_id = args.document_id
    print(f"[compare] workspace={workspace_dir}", flush=True)
    print(f"[compare] document_id={document_id} chunk_index={args.chunk_index}", flush=True)

    doc2chunks_path = workspace_dir / "knowledge_graph" / "doc2chunks.json"
    extraction_results_path = workspace_dir / "knowledge_graph" / "extraction_results.json"
    doc2chunks = _load_json(doc2chunks_path)
    extraction_results = _load_json(extraction_results_path)

    if document_id not in doc2chunks:
        raise KeyError(f"document_id not found in doc2chunks: {document_id}")
    if document_id not in extraction_results:
        raise KeyError(f"document_id not found in extraction_results: {document_id}")

    chunk_record = doc2chunks[document_id]
    result_record = extraction_results[document_id]
    chunks = chunk_record.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"no chunks found for document_id={document_id}")
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise IndexError(f"chunk_index out of range: {args.chunk_index}, available=0..{len(chunks)-1}")
    selected_chunk = chunks[args.chunk_index]
    text = str(selected_chunk.get("content", "")).strip()
    print(
        f"[compare] selected_chunk_id={selected_chunk.get('id', '')} text_words={len(text.split())} entities={len(result_record.get('entities', []))}",
        flush=True,
    )
    if args.aggressive_clean:
        text = clean_screenplay_text(text, aggressive=True)
    entities = copy.deepcopy(result_record.get("entities", []))
    if not isinstance(entities, list) or not entities:
        raise ValueError(f"no entities found for document_id={document_id}")

    config = load_config(str(Path(args.config).resolve()))
    config.llm.max_tokens = min(int(getattr(config.llm, "max_tokens", 8192) or 8192), 3072)
    config.llm.timeout = int(args.llm_timeout)
    if args.llm_base_url:
        config.llm.base_url = str(args.llm_base_url).strip()
    if args.llm_model_name:
        config.llm.model_name = str(args.llm_model_name).strip()
    print(f"[compare] config_language={config.global_config.language} schema_dir={config.global_config.schema_dir}", flush=True)
    print(f"[compare] llm_model={config.llm.model_name} llm_base_url={config.llm.base_url}", flush=True)
    entity_schema_path = repo_root / config.global_config.schema_dir / "default_entity_schema.json"
    relation_schema_path = repo_root / config.global_config.schema_dir / "default_relation_schema.json"
    entity_schema = _load_json(entity_schema_path)
    relation_schema = _load_json(relation_schema_path)
    llm = OpenAILLM(
        config,
        timeout=int(getattr(config.llm, "timeout", 180) or 180),
        max_retries=0,
    )
    agent = InformationExtractionAgent(
        config=config,
        llm=llm,
        entity_schema=entity_schema,
        relation_schema=relation_schema,
        memory_store=None,
    )
    print("[compare] agent initialized", flush=True)

    old_run = _run_mode(
        agent=agent,
        mode=args.old_mode,
        text=text,
        entities=entities,
        rid_namespace=f"{document_id}:{args.old_mode}",
    )
    new_run = _run_mode(
        agent=agent,
        mode=args.new_mode,
        text=text,
        entities=entities,
        rid_namespace=f"{document_id}:{args.new_mode}",
    )

    old_summary = _summarize_mode(
        mode=args.old_mode,
        raw_relations=old_run["raw_relations"],
        feedbacks_before_fix=old_run["feedbacks_before_fix"],
        fixed_relations=old_run["fixed_relations_before_dedup"],
        final_relations=old_run["final_relations_after_dedup"],
        entities=entities,
    )
    new_summary = _summarize_mode(
        mode=args.new_mode,
        raw_relations=new_run["raw_relations"],
        feedbacks_before_fix=new_run["feedbacks_before_fix"],
        fixed_relations=new_run["fixed_relations_before_dedup"],
        final_relations=new_run["final_relations_after_dedup"],
        entities=entities,
    )

    final_compare = _compare_sets(
        old_run["final_relations_after_dedup"],
        new_run["final_relations_after_dedup"],
    )
    raw_compare = _compare_sets(
        old_run["raw_relations"],
        new_run["raw_relations"],
    )

    summary = {
        "workspace_dir": str(workspace_dir),
        "document_id": document_id,
        "chunk_index": args.chunk_index,
        "chunk_id": selected_chunk.get("id", ""),
        "config": str(Path(args.config).resolve()),
        "old_mode": args.old_mode,
        "new_mode": args.new_mode,
        "text_word_count": len(text.split()),
        "entity_count": len(entities),
        "chunk_count": len(chunks),
        "old_summary": old_summary,
        "new_summary": new_summary,
        "raw_compare": raw_compare,
        "final_compare": final_compare,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(output_dir / "input_meta.json", {
        "workspace_dir": str(workspace_dir),
        "document_id": document_id,
        "available_chunk_ids": [chunk.get("id", "") for chunk in chunks],
        "selected_chunk_index": args.chunk_index,
        "selected_chunk_id": selected_chunk.get("id", ""),
        "available_chunk_count": len(chunks),
        "entity_count": len(entities),
        "text_word_count": len(text.split()),
    })
    _dump_json(output_dir / "frozen_entities.json", entities)
    _dump_json(output_dir / "old_relations_raw.json", old_run["raw_relations"])
    _dump_json(output_dir / "old_feedbacks_before_fix.json", old_run["feedbacks_before_fix"])
    _dump_json(output_dir / "old_relations_fixed_before_dedup.json", old_run["fixed_relations_before_dedup"])
    _dump_json(output_dir / "old_relations_final.json", old_run["final_relations_after_dedup"])
    _dump_json(output_dir / "new_relations_raw.json", new_run["raw_relations"])
    _dump_json(output_dir / "new_feedbacks_before_fix.json", new_run["feedbacks_before_fix"])
    _dump_json(output_dir / "new_relations_fixed_before_dedup.json", new_run["fixed_relations_before_dedup"])
    _dump_json(output_dir / "new_relations_final.json", new_run["final_relations_after_dedup"])
    _dump_json(output_dir / "comparison_summary.json", summary)
    (output_dir / "comparison_report.md").write_text(
        _render_markdown_report(
            workspace_dir=workspace_dir,
            document_id=document_id,
            chunk_index=args.chunk_index,
            chunk_id=str(selected_chunk.get("id", "")),
            text_word_count=len(text.split()),
            entity_count=len(entities),
            old_summary=old_summary,
            new_summary=new_summary,
            final_compare=final_compare,
        ),
        encoding="utf-8",
    )

    print(f"[compare] artifacts_saved={output_dir}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
