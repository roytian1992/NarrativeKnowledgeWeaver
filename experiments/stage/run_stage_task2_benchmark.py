from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.builder.graph_builder import KnowledgeGraphBuilder
from core.builder.narrative_graph_builder import NarrativeGraphBuilder
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.model_providers.openai_llm import OpenAILLM
from core.storage.graph_store import GraphStore
from core.utils.config import KAGConfig, _apply_global_locale_paths
from core.utils.general_utils import json_dump_atomic
from core.utils.prompt_loader import YAMLPromptLoader
from experiments.quality.run_quality_benchmark import (
    _article_interaction_artifact_paths,
    _ensure_runtime_graph_embedding_cache,
    _ensure_workspace_vector_stores_current,
    _existing_workspace_runtime_incomplete,
    _extract_json_object,
    _extract_llm_text,
    _extract_semantic_answer_text,
    _has_error_payload,
    _load_existing_article_to_graph_store,
    _looks_like_tool_call_payload,
    _resolve_article_workspace_dir,
    _resolve_cli_path,
    _runtime_graph_has_data,
    _summarize_setting,
    _summarize_tool_uses_for_finalization,
    _update_workspace_asset_registry,
    _workspace_missing_artifacts,
    _write_setting_progress,
)

logger = logging.getLogger(__name__)


DEFAULT_RETRIEVAL_PROFILE = "default"
HYBRID_RAG_RETRIEVAL_PROFILE = "hybrid_rag"

_BASE_HIDDEN_TOOL_NAMES: set[str] = {
    "get_co_section_entities",
    "get_k_hop_subgraph",
    "get_common_neighbors",
    "get_relations_between_entities",
    "find_paths_between_nodes",
    "vdb_search_docs",
    "search_related_content",
    "choice_grounded_evidence_search",
    "community_graphrag_search",
    "search_communities",
    "implication_constrained_inference_search",
}

_HYBRID_RAG_ALLOWED_TOOL_NAMES: set[str] = {
    "bm25_search_docs",
    "vdb_search_docs",
    "vdb_search_sentences",
    "vdb_get_docs_by_document_ids",
    "lookup_titles_by_document_ids",
    "lookup_document_ids_by_title",
}


def _normalize_retrieval_profile(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    if not value:
        return DEFAULT_RETRIEVAL_PROFILE
    if value in {"default", "no_strategy_agent", "full"}:
        return DEFAULT_RETRIEVAL_PROFILE
    if value in {"hybrid", "hybrid_rag", "hybrid-rag", "rag_hybrid"}:
        return HYBRID_RAG_RETRIEVAL_PROFILE
    raise ValueError(f"Unsupported retrieval profile: {raw_value}")


def _resolve_workspace_build_mode(raw_value: str, *, retrieval_profile: str) -> str:
    value = str(raw_value or "").strip().lower()
    if not value or value == "auto":
        if retrieval_profile == HYBRID_RAG_RETRIEVAL_PROFILE:
            return "hybrid_rag_light"
        return "full"
    if value in {"full", "hybrid_rag_light"}:
        return value
    raise ValueError(f"Unsupported workspace build mode: {raw_value}")


def _setting_name_for_profile(retrieval_profile: str) -> str:
    if retrieval_profile == HYBRID_RAG_RETRIEVAL_PROFILE:
        return HYBRID_RAG_RETRIEVAL_PROFILE
    return "no_strategy_agent"


def _profile_uses_sql_tools(retrieval_profile: str, *, cli_enable_sql_tools: bool) -> bool:
    if retrieval_profile == HYBRID_RAG_RETRIEVAL_PROFILE:
        return False
    return bool(cli_enable_sql_tools)


def _tool_allow_list_for_profile(retrieval_profile: str) -> Optional[set[str]]:
    if retrieval_profile == HYBRID_RAG_RETRIEVAL_PROFILE:
        return set(_HYBRID_RAG_ALLOWED_TOOL_NAMES)
    return None


def _build_stage_signature(
    cfg: KAGConfig,
    *,
    retrieval_profile: str,
    build_mode: str,
) -> Dict[str, Any]:
    return {
        "retrieval_profile": str(retrieval_profile or DEFAULT_RETRIEVAL_PROFILE),
        "build_mode": str(build_mode or "full"),
        "chunk_size": int(getattr(cfg.document_processing, "chunk_size", 0) or 0),
        "chunk_overlap": int(getattr(cfg.document_processing, "chunk_overlap", 0) or 0),
        "sentence_chunk_size": int(getattr(cfg.document_processing, "sentence_chunk_size", 0) or 0),
        "sentence_chunk_overlap": int(getattr(cfg.document_processing, "sentence_chunk_overlap", 0) or 0),
        "bm25_chunk_size": int(getattr(cfg.document_processing, "bm25_chunk_size", 0) or 0),
    }


def _read_stage_build_marker(marker_path: Path) -> Dict[str, Any]:
    if not marker_path.exists():
        return {}
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_build_marker_matches(marker_payload: Dict[str, Any], *, signature: Dict[str, Any]) -> bool:
    if not isinstance(marker_payload, dict) or not marker_payload:
        return False

    has_signature = any(
        key in marker_payload
        for key in (
            "retrieval_profile",
            "build_mode",
            "chunk_size",
            "chunk_overlap",
            "sentence_chunk_size",
            "sentence_chunk_overlap",
            "bm25_chunk_size",
        )
    )
    if not has_signature:
        return (
            str(signature.get("retrieval_profile", DEFAULT_RETRIEVAL_PROFILE)) == DEFAULT_RETRIEVAL_PROFILE
            and str(signature.get("build_mode", "full")) == "full"
        )

    for key, expected in signature.items():
        if marker_payload.get(key) != expected:
            return False
    return True


def _stage_workspace_missing_artifacts(workspace_dir: Path, *, build_mode: str) -> List[str]:
    required_paths = [
        workspace_dir / "build_marker.json",
        workspace_dir / "knowledge_graph" / "all_document_chunks.json",
        workspace_dir / "knowledge_graph" / "doc2chunks.json",
        workspace_dir / "knowledge_graph" / "doc2chunks_index.json",
    ]
    if build_mode == "full":
        required_paths.extend(
            [
                workspace_dir / "interactions" / "interaction_results.json",
                workspace_dir / "interactions" / "interaction_records_list.json",
                workspace_dir / "sql" / "Interaction.db",
            ]
        )

    missing: List[str] = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))
    return missing


@dataclass
class StageEvalResult:
    setting: str
    article_name: str
    run_index: int
    question_id: str
    question: str
    reference_answer: str
    predicted_answer: str
    is_correct: bool
    latency_ms: int
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting": self.setting,
            "article_name": self.article_name,
            "run_index": self.run_index,
            "question_id": self.question_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            **self.extra,
        }


def _default_workspace_asset_root() -> Path:
    return REPO_ROOT / "experiments" / "stage" / "assets" / "article_workspaces"


def _default_converted_script_root() -> Path:
    return REPO_ROOT / "experiments" / "stage" / "assets" / "converted_scripts"


def _set_global_language(cfg: KAGConfig, language: str) -> None:
    cfg.global_.language = language
    cfg.global_.locale = language
    cfg.global_config.language = language
    cfg.global_config.locale = language
    _apply_global_locale_paths(cfg.global_)
    _apply_global_locale_paths(cfg.global_config)


def _build_stage_article_config(
    base_cfg: KAGConfig,
    *,
    workspace_dir: Path,
    language: str,
    build_max_workers: int,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    sentence_chunk_size: Optional[int] = None,
    sentence_chunk_overlap: Optional[int] = None,
    bm25_chunk_size: Optional[int] = None,
    strategy_enabled: bool = False,
    subagent_enabled: bool = False,
) -> KAGConfig:
    cfg = copy.deepcopy(base_cfg)
    _set_global_language(cfg, language)
    cfg.global_.doc_type = "screenplay"
    cfg.global_config.doc_type = "screenplay"
    cfg.global_.aggregation_mode = "narrative"
    cfg.global_config.aggregation_mode = "narrative"

    effective_build_workers = max(1, int(build_max_workers or cfg.knowledge_graph_builder.max_workers or 16))
    cfg.document_processing.max_workers = effective_build_workers
    cfg.document_processing.reset_vector_collections = True
    cfg.knowledge_graph_builder.max_workers = effective_build_workers
    cfg.narrative_graph_builder.max_workers = effective_build_workers

    if chunk_size is not None and int(chunk_size) > 0:
        cfg.document_processing.chunk_size = int(chunk_size)
        if sentence_chunk_size is None:
            sentence_chunk_size = int(chunk_size)
        if bm25_chunk_size is None:
            bm25_chunk_size = int(chunk_size)
    if chunk_overlap is not None and int(chunk_overlap) >= 0:
        cfg.document_processing.chunk_overlap = int(chunk_overlap)
        if sentence_chunk_overlap is None:
            sentence_chunk_overlap = int(chunk_overlap)
    if sentence_chunk_size is not None and int(sentence_chunk_size) > 0:
        cfg.document_processing.sentence_chunk_size = int(sentence_chunk_size)
    if sentence_chunk_overlap is not None and int(sentence_chunk_overlap) >= 0:
        cfg.document_processing.sentence_chunk_overlap = int(sentence_chunk_overlap)
    if bm25_chunk_size is not None and int(bm25_chunk_size) > 0:
        cfg.document_processing.bm25_chunk_size = int(bm25_chunk_size)

    cfg.knowledge_graph_builder.file_path = str(workspace_dir / "knowledge_graph")
    cfg.narrative_graph_builder.file_path = str(workspace_dir / "narrative_graph")
    community_dir = workspace_dir / "community_graph"
    cfg.community_graph_builder.file_path = str(community_dir) if community_dir.exists() else ""
    cfg.storage.graph_store_path = str(workspace_dir / "knowledge_graph" / "graph_runtime_langgraph.pkl")
    cfg.storage.vector_store_path = str(workspace_dir / "vector_store")
    cfg.storage.sql_database_path = str(workspace_dir / "sql")

    cfg.extraction_memory.enabled = False
    cfg.strategy_memory.enabled = strategy_enabled
    cfg.strategy_memory.read_enabled = bool(strategy_enabled)
    cfg.strategy_memory.subagent_enabled = subagent_enabled
    cfg.strategy_memory.online_runtime_mode = False
    return cfg


def _build_stage_article_workspace(
    cfg: KAGConfig,
    json_file_path: Path,
    *,
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE,
    build_mode: str = "full",
) -> Dict[str, Any]:
    interaction_paths = _article_interaction_artifact_paths(cfg)
    interaction_paths["interaction_dir"].mkdir(parents=True, exist_ok=True)
    interaction_paths["sql_db_path"].parent.mkdir(parents=True, exist_ok=True)
    kg_workers = max(1, int(cfg.knowledge_graph_builder.max_workers or 16))
    ng_workers = max(1, int(cfg.narrative_graph_builder.max_workers or 16))

    builder = KnowledgeGraphBuilder(cfg, use_memory=False)
    try:
        builder.prepare_chunks(json_file_path=str(json_file_path), reset_output_dir=True, reset_vector_collections=True)
        builder.extract_entity_and_relation(
            retries=3,
            concurrency=kg_workers,
            per_task_timeout=int(cfg.knowledge_graph_builder.per_task_timeout or 2400),
            reset_outputs=True,
        )
        builder.run_extraction_refinement()
        builder.build_entity_and_relation_basic_info()
        builder.postprocess_and_save()
        builder.extract_properties()
        builder.extract_interactions(
            retries=3,
            concurrency=kg_workers,
            per_task_timeout=int(cfg.knowledge_graph_builder.per_task_timeout or 2400),
            interaction_json_path=str(interaction_paths["interaction_json_path"]),
            interaction_list_json_path=str(interaction_paths["interaction_list_json_path"]),
        )
        builder.store_interactions_to_sql(
            interaction_list_json_path=str(interaction_paths["interaction_list_json_path"]),
            sql_db_path=str(interaction_paths["sql_db_path"]),
            reset_database=True,
            reset_table=True,
        )
        builder.build_doc_entities()
        builder.load_json_to_graph_store()
    finally:
        try:
            builder.graph_store.close()
        except Exception:
            pass

    narrative_builder = NarrativeGraphBuilder(cfg)
    try:
        narrative_builder.extract_episodes(
            limit_documents=200000,
            document_concurrency=ng_workers,
            store_episode_support_edges=True,
            ensure_episode_embeddings=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        narrative_builder.extract_episode_relations(
            episode_pair_concurrency=ng_workers,
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
        narrative_builder.break_episode_cycles(method="saber")
        narrative_builder.build_storyline_candidates(method="trie", min_trunk_len=2)
        narrative_builder.extract_storylines_from_candidates(
            ensure_storyline_embeddings=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        narrative_builder.extract_storyline_relations(
            max_storyline_pairs_global=500,
            similarity_threshold=0.5,
            overlap_pair_only=True,
            min_shared_anchor_count=2,
            show_pair_progress=False,
            storyline_pair_concurrency=ng_workers,
        )
        narrative_builder.load_json_to_graph_store(
            store_episodes=True,
            store_support_edges=True,
            store_episode_relations=True,
            store_storylines=True,
            store_storyline_support_edges=True,
            store_storyline_relations=True,
        )
    finally:
        try:
            narrative_builder.graph_store.close()
        except Exception:
            pass

    return _write_stage_build_marker(
        cfg,
        json_file_path,
        retrieval_profile=retrieval_profile,
        build_mode=build_mode,
    )


def _build_stage_article_workspace_hybrid_rag_light(
    cfg: KAGConfig,
    json_file_path: Path,
    *,
    retrieval_profile: str = HYBRID_RAG_RETRIEVAL_PROFILE,
    build_mode: str = "hybrid_rag_light",
) -> Dict[str, Any]:
    builder = KnowledgeGraphBuilder(cfg, use_memory=False)
    try:
        builder.prepare_chunks(json_file_path=str(json_file_path), reset_output_dir=True, reset_vector_collections=True)
    finally:
        try:
            builder.graph_store.close()
        except Exception:
            pass

    graph_store = GraphStore(cfg)
    try:
        graph_store.close()
    except Exception:
        pass

    return _write_stage_build_marker(
        cfg,
        json_file_path,
        retrieval_profile=retrieval_profile,
        build_mode=build_mode,
    )


def _write_stage_build_marker(
    cfg: KAGConfig,
    json_file_path: Path,
    *,
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE,
    build_mode: str = "full",
) -> Dict[str, Any]:
    interaction_paths = _article_interaction_artifact_paths(cfg)
    marker = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "json_file_path": str(json_file_path),
        "knowledge_graph_dir": cfg.knowledge_graph_builder.file_path,
        "narrative_graph_dir": cfg.narrative_graph_builder.file_path,
        "interaction_json_path": str(interaction_paths["interaction_json_path"]),
        "interaction_list_json_path": str(interaction_paths["interaction_list_json_path"]),
        "interaction_sql_db_path": str(interaction_paths["sql_db_path"]),
        **_build_stage_signature(cfg, retrieval_profile=retrieval_profile, build_mode=build_mode),
    }
    marker_path = Path(cfg.knowledge_graph_builder.file_path).parent / "build_marker.json"
    json_dump_atomic(str(marker_path), marker)
    return marker


def _resume_stage_article_workspace(
    cfg: KAGConfig,
    *,
    json_file_path: Path,
    build_max_workers: int,
    retrieval_profile: str = DEFAULT_RETRIEVAL_PROFILE,
    build_mode: str = "full",
) -> bool:
    workspace_dir = Path(str(cfg.knowledge_graph_builder.file_path)).parent
    interaction_paths = _article_interaction_artifact_paths(cfg)
    narrative_global_dir = workspace_dir / "narrative_graph" / "global"
    required_resume_paths = [
        workspace_dir / "knowledge_graph" / "doc2chunks.json",
        interaction_paths["interaction_json_path"],
        interaction_paths["interaction_list_json_path"],
        interaction_paths["sql_db_path"],
        narrative_global_dir / "episodes.json",
        narrative_global_dir / "episode_relations.json",
        narrative_global_dir / "episode_relations_dag.json",
        narrative_global_dir / "storylines.json",
        narrative_global_dir / "storyline_support_edges.json",
    ]
    missing_resume_paths = [str(path) for path in required_resume_paths if not path.exists()]
    if missing_resume_paths:
        logger.info(
            "[STAGE][Workspace] partial resume unavailable for %s, missing=%s",
            workspace_dir.name,
            missing_resume_paths,
        )
        return False

    storyline_rel_path = narrative_global_dir / "storyline_relations.json"
    need_storyline_relations = not storyline_rel_path.exists()
    need_runtime_reload = (not _runtime_graph_has_data(cfg)) or _existing_workspace_runtime_incomplete(cfg)
    need_marker = not (workspace_dir / "build_marker.json").exists()

    if not (need_storyline_relations or need_runtime_reload or need_marker):
        return True

    logger.info(
        "[STAGE][Workspace] partial resume start: movie=%s need_storyline_relations=%s need_runtime_reload=%s need_marker=%s",
        workspace_dir.name,
        "yes" if need_storyline_relations else "no",
        "yes" if need_runtime_reload else "no",
        "yes" if need_marker else "no",
    )

    narrative_builder = NarrativeGraphBuilder(cfg)
    try:
        if need_storyline_relations:
            narrative_builder.extract_storyline_relations(
                max_storyline_pairs_global=500,
                similarity_threshold=0.5,
                overlap_pair_only=True,
                min_shared_anchor_count=2,
                show_pair_progress=False,
                storyline_pair_concurrency=max(1, int(build_max_workers or cfg.narrative_graph_builder.max_workers or 16)),
            )

        if need_storyline_relations or need_runtime_reload:
            narrative_builder.load_json_to_graph_store(
                store_episodes=True,
                store_support_edges=True,
                store_episode_relations=True,
                store_storylines=True,
                store_storyline_support_edges=True,
                store_storyline_relations=storyline_rel_path.exists(),
            )
    finally:
        try:
            narrative_builder.graph_store.close()
        except Exception:
            pass

    _write_stage_build_marker(
        cfg,
        json_file_path,
        retrieval_profile=retrieval_profile,
        build_mode=build_mode,
    )
    logger.info("[STAGE][Workspace] partial resume finished: movie=%s", workspace_dir.name)
    return True


def _ensure_stage_article_ready(
    *,
    base_cfg: KAGConfig,
    article_json_path: Path,
    workspace_dir: Path,
    language: str,
    build_max_workers: int,
    retrieval_profile: str,
    workspace_build_mode: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    sentence_chunk_size: Optional[int] = None,
    sentence_chunk_overlap: Optional[int] = None,
    bm25_chunk_size: Optional[int] = None,
    rebuild: bool = False,
) -> KAGConfig:
    cfg = _build_stage_article_config(
        base_cfg,
        workspace_dir=workspace_dir,
        language=language,
        build_max_workers=build_max_workers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sentence_chunk_size=sentence_chunk_size,
        sentence_chunk_overlap=sentence_chunk_overlap,
        bm25_chunk_size=bm25_chunk_size,
        strategy_enabled=False,
        subagent_enabled=False,
    )
    marker_path = workspace_dir / "build_marker.json"
    signature = _build_stage_signature(cfg, retrieval_profile=retrieval_profile, build_mode=workspace_build_mode)
    marker_payload = _read_stage_build_marker(marker_path)
    marker_matches = _stage_build_marker_matches(marker_payload, signature=signature)
    if marker_path.exists() and marker_matches and not rebuild and not _stage_workspace_missing_artifacts(workspace_dir, build_mode=workspace_build_mode):
        if workspace_build_mode == "hybrid_rag_light":
            _ensure_workspace_vector_stores_current(cfg)
            return cfg
        if _runtime_graph_has_data(cfg) and not _existing_workspace_runtime_incomplete(cfg):
            _ensure_runtime_graph_embedding_cache(cfg)
            return cfg
        _load_existing_article_to_graph_store(cfg)
        return cfg
    if workspace_dir.exists() and not rebuild and workspace_build_mode == "full" and marker_matches:
        if _resume_stage_article_workspace(
            cfg,
            json_file_path=article_json_path,
            build_max_workers=build_max_workers,
            retrieval_profile=retrieval_profile,
            build_mode=workspace_build_mode,
        ):
            if _runtime_graph_has_data(cfg) and not _existing_workspace_runtime_incomplete(cfg):
                _ensure_runtime_graph_embedding_cache(cfg)
            else:
                _load_existing_article_to_graph_store(cfg)
            return cfg
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if workspace_build_mode == "hybrid_rag_light":
        _build_stage_article_workspace_hybrid_rag_light(
            cfg,
            article_json_path,
            retrieval_profile=retrieval_profile,
            build_mode=workspace_build_mode,
        )
    else:
        _build_stage_article_workspace(
            cfg,
            article_json_path,
            retrieval_profile=retrieval_profile,
            build_mode=workspace_build_mode,
        )
    return cfg


def _prepare_stage_article_cfg(
    *,
    base_cfg: KAGConfig,
    article_name: str,
    article_json_path: Path,
    workspace_dir: Path,
    workspace_asset_root: Optional[Path],
    language: str,
    build_max_workers: int,
    retrieval_profile: str,
    workspace_build_mode: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    sentence_chunk_size: Optional[int] = None,
    sentence_chunk_overlap: Optional[int] = None,
    bm25_chunk_size: Optional[int] = None,
    rebuild: bool = False,
) -> KAGConfig:
    cfg = _ensure_stage_article_ready(
        base_cfg=base_cfg,
        article_json_path=article_json_path,
        workspace_dir=workspace_dir,
        language=language,
        build_max_workers=build_max_workers,
        retrieval_profile=retrieval_profile,
        workspace_build_mode=workspace_build_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sentence_chunk_size=sentence_chunk_size,
        sentence_chunk_overlap=sentence_chunk_overlap,
        bm25_chunk_size=bm25_chunk_size,
        rebuild=rebuild,
    )
    _ensure_workspace_vector_stores_current(cfg)
    _update_workspace_asset_registry(
        workspace_asset_root=workspace_asset_root,
        article_name=article_name,
        workspace_dir=workspace_dir,
        article_json_path=article_json_path,
    )
    return cfg


def _normalize_stage_scene(
    *,
    movie_id: str,
    language: str,
    item: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    content = str(item.get("content", "") or "").strip()
    if not content:
        return None
    raw_scene_id = item.get("_id")
    if raw_scene_id is None:
        raw_scene_id = item.get("id")
    scene_id = str(raw_scene_id).strip() if raw_scene_id is not None and str(raw_scene_id).strip() else str(index)
    title = str(item.get("title", "") or "").strip() or f"Scene {scene_id}"
    subtitle = str(item.get("subtitle", "") or "").strip()
    return {
        "id": f"{movie_id}_scene_{scene_id}",
        "title": title,
        "subtitle": subtitle,
        "content": content,
        "metadata": {
            "movie_id": movie_id,
            "language": language,
            "source_scene_id": scene_id,
            "source_scene_title": title,
        },
    }


def _convert_stage_script(
    *,
    movie_id: str,
    language: str,
    src_path: Path,
    dst_path: Path,
) -> None:
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected script.json list: {src_path}")
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_stage_scene(movie_id=movie_id, language=language, item=item, index=index)
        if normalized is not None:
            out.append(normalized)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump_atomic(str(dst_path), out)


def _split_related_scenes(raw_value: Any) -> List[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    normalized = (
        text.replace("；", ";")
        .replace("\n", ";")
        .replace(" , ", ";")
        .replace("，", ";")
    )
    parts = [piece.strip() for piece in normalized.split(";")]
    seen = set()
    out: List[str] = []
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def _format_stage_reference_answers(answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return "(no reference answer)"
    return "\n".join(
        [
            "The following are acceptable reference answers or answer variants:",
            f"- {cleaned}",
        ]
    )


def _load_stage_task2_qas(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str((row or {}).get("question", "") or "").strip()
            answer = str((row or {}).get("answer", "") or "").strip()
            if not question:
                continue
            language = str((row or {}).get("language", "") or "").strip().lower() or "en"
            question_type = str((row or {}).get("question_type", "") or "").strip()
            related_scenes = _split_related_scenes((row or {}).get("related_scenes", ""))
            evidence_or_reason = str((row or {}).get("evidence_or_reason", "") or "").strip()
            metadata = dict(row or {})
            rows.append(
                {
                    "question_id": str((row or {}).get("id", "") or "").strip() or f"q{len(rows)}",
                    "question": question,
                    "reference_answers": [answer] if answer else [],
                    "reference_answer": _format_stage_reference_answers(answer),
                    "metadata": metadata,
                    "language": language,
                    "question_type": question_type,
                    "question_type_tags": [question_type] if question_type else [],
                    "related_scenes": related_scenes,
                    "evidence_or_reason": evidence_or_reason,
                }
            )
    return rows


def _format_open_question_for_agent(question_text: str, *, lang: str) -> str:
    stem = str(question_text or "").strip()
    if lang == "zh":
        lines = [
            "请使用检索来回答下面的问题。",
            "",
            "只返回 JSON：",
            '{"answer_text":"...","evidence":"...","confidence":0.72}',
            "",
            "规则：",
            "- `answer_text` 必须直接回答问题，用一句简短的话或短语。",
            "- `evidence` 必须简短，并且基于检索到的证据。",
            "- `confidence` 必须是 0 到 1 之间的数字。",
            "- 不要输出选项标签。",
            "- 不要输出推理过程、计划或多余解释。",
            "",
            "问题：",
            stem,
        ]
        return "\n".join(lines).strip()
    lines = [
        "Answer the following question using retrieval.",
        "",
        "Return JSON only:",
        '{"answer_text":"...","evidence":"...","confidence":0.72}',
        "",
        "Rules:",
        "- `answer_text` should directly answer the question in one short sentence or phrase.",
        "- `evidence` should be brief and grounded in retrieved evidence.",
        "- `confidence` must be a number between 0 and 1.",
        "- Do not output option labels.",
        "- Do not output a plan, chain-of-thought, or extra explanation.",
        "",
        "Question:",
        stem,
    ]
    return "\n".join(lines).strip()


def _format_open_question_for_agent_with_retrieval_guard(question_text: str, *, lang: str) -> str:
    base = _format_open_question_for_agent(question_text, lang=lang)
    if lang == "zh":
        extra = "\n".join(
            [
                "",
                "额外要求：",
                "- 先检索，再作答。",
                "- 如果当前证据不够，请继续检索，不要直接猜测。",
                "- 如果答案涉及场景、人物关系、动机或时间顺序，请优先检索能区分这些信息的证据。",
            ]
        )
        return base + extra
    extra = "\n".join(
        [
            "",
            "Additional requirements:",
            "- Retrieve evidence before answering.",
            "- If current evidence is insufficient, retrieve more instead of guessing.",
            "- For scenes, relations, motives, or temporal order, prioritize retrieval that distinguishes those details.",
        ]
    )
    return base + extra


def _ensure_open_answer_payload(answer_text: str, *, default_confidence: float = 0.62) -> str:
    raw = str(answer_text or "").strip()
    if not raw:
        return raw
    payload = _extract_json_object(raw)
    if isinstance(payload, dict):
        answer_value = str(
            payload.get("answer_text")
            or payload.get("answer")
            or payload.get("final_answer")
            or ""
        ).strip()
        if answer_value:
            payload["answer_text"] = answer_value
            if payload.get("confidence") is None:
                payload["confidence"] = float(default_confidence)
            if payload.get("evidence") is None:
                payload["evidence"] = ""
            return json.dumps(payload, ensure_ascii=False)
    semantic = _extract_semantic_answer_text(raw)
    if not semantic:
        semantic = raw
    return json.dumps(
        {
            "answer_text": semantic,
            "evidence": "",
            "confidence": float(default_confidence),
        },
        ensure_ascii=False,
    )


class OpenAgentThreadLocal:
    def __init__(
        self,
        cfg: KAGConfig,
        *,
        setting_name: str,
        retrieval_profile: str,
        article_language: str,
        enable_sql_tools: bool,
    ) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.retrieval_profile = _normalize_retrieval_profile(retrieval_profile)
        self.article_language = str(article_language or "").strip().lower() or "en"
        self.enable_sql_tools = bool(enable_sql_tools)
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        extra_hidden_tool_names = {
            str(name or "").strip()
            for name in (getattr(getattr(cfg, "strategy_memory", None), "hidden_tool_names", []) or [])
            if str(name or "").strip()
        }
        self.base_hidden_tool_names = set(_BASE_HIDDEN_TOOL_NAMES)
        self.extra_hidden_tool_names = extra_hidden_tool_names
        self.allowed_tool_names = _tool_allow_list_for_profile(self.retrieval_profile)

    def _build_state(self) -> Dict[str, Any]:
        agent = QuestionAnsweringAgent(
            self.cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
        )
        hidden_tool_names = self.base_hidden_tool_names | self.extra_hidden_tool_names
        if self.allowed_tool_names is not None:
            all_tool_names = {
                str(getattr(tool, "name", "") or "").strip()
                for tool in [*getattr(agent, "_base_tools", []), *getattr(agent, "_aggregation_tools", []), *getattr(agent, "_extra_tools", [])]
                if str(getattr(tool, "name", "") or "").strip()
            }
            hidden_tool_names = {
                name
                for name in all_tool_names
                if name not in self.allowed_tool_names
            } | {
                name
                for name in self.extra_hidden_tool_names
                if name not in self.allowed_tool_names
            }
        setattr(agent, "hidden_tool_names", set(hidden_tool_names))
        rebuild = getattr(agent, "_rebuild_assistant", None)
        if callable(rebuild):
            rebuild()
        prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        finalizer_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge = RetrievalAnswerJudge(
            prompt_loader=prompt_loader,
            llm=judge_llm,
            prompt_id="memory/judge_open_retrieval_answer",
        )
        state = {
            "agent": agent,
            "finalizer_llm": finalizer_llm,
            "judge": judge,
        }
        with self._lock:
            self._states.append(state)
        return state

    def state(self) -> Dict[str, Any]:
        state = getattr(self.local, "state", None)
        if state is None:
            state = self._build_state()
            self.local.state = state
        return state

    def close(self) -> None:
        for state in self._states:
            try:
                state["agent"].close()
            except Exception:
                pass


def _trim_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


class OpenAgentEvaluator:
    def __init__(
        self,
        cfg: KAGConfig,
        *,
        setting_name: str,
        retrieval_profile: str,
        article_name: str,
        article_language: str,
        enable_sql_tools: bool,
    ) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.article_name = article_name
        self.article_language = str(article_language or "").strip().lower() or "en"
        self.tlocal = OpenAgentThreadLocal(
            cfg,
            setting_name=setting_name,
            retrieval_profile=retrieval_profile,
            article_language=self.article_language,
            enable_sql_tools=enable_sql_tools,
        )

    def _repair_open_answer(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, Any],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        if not tool_uses:
            return current_answer
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses, max_items=8)
        if not evidence_block:
            return current_answer
        if self.article_language == "zh":
            prompt = "\n".join(
                [
                    "你是开放问答的最终答案整理器。",
                    "你的任务是把检索到的证据和当前不完整答案整理成一个最终答案。",
                    "只返回 JSON：",
                    '{"answer_text":"...","evidence":"...","confidence":0.72}',
                    "",
                    "规则：",
                    "- 直接回答问题，使用一句简短的话或短语。",
                    "- 只能使用下方检索到的证据。",
                    "- `evidence` 必须简短且有依据。",
                    "- `confidence` 必须是 0 到 1 之间的数字。",
                    "- 如果证据有歧义，保守回答，不要编造细节。",
                    "- 不要返回工具调用、计划或中间分析。",
                    "",
                    "问题：",
                    str(row.get("question", "") or ""),
                    "",
                    "检索到的证据：",
                    evidence_block,
                    "",
                    "之前的答案或未完成草稿：",
                    _trim_text(current_answer, limit=700) or "(none)",
                ]
            )
        else:
            prompt = "\n".join(
                [
                    "You are the final open-question answer adapter.",
                    "Your job is to convert retrieved evidence and any incomplete prior answer into one grounded final answer.",
                    "Return JSON only:",
                    '{"answer_text":"...","evidence":"...","confidence":0.72}',
                    "",
                    "Rules:",
                    "- Answer the question directly in one short sentence or phrase.",
                    "- Use only the retrieved evidence below.",
                    "- `evidence` must be brief and grounded.",
                    "- `confidence` must be a number between 0 and 1.",
                    "- If the evidence is ambiguous, answer conservatively instead of inventing details.",
                    "- Do not return a tool call, a plan, or an intermediate analysis.",
                    "",
                    "Question:",
                    str(row.get("question", "") or ""),
                    "",
                    "Retrieved evidence:",
                    evidence_block,
                    "",
                    "Previous answer or unfinished draft:",
                    _trim_text(current_answer, limit=700) or "(none)",
                ]
            )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            adapted = _extract_llm_text(result)
            semantic = _extract_semantic_answer_text(adapted)
            if semantic and not _looks_like_tool_call_payload(adapted):
                return _ensure_open_answer_payload(adapted)
        except Exception:
            return current_answer
        return current_answer

    def evaluate_row(self, row: Dict[str, Any], run_index: int) -> StageEvalResult:
        started = time.time()
        state = self.tlocal.state()
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        raw_agent_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        strategy_context: Dict[str, Any] = {}
        error_text = ""
        prompt_variants = [
            (
                "base",
                _format_open_question_for_agent(str(row.get("question", "") or ""), lang=self.article_language),
                (8, 10, 12),
            ),
            (
                "guarded",
                _format_open_question_for_agent_with_retrieval_guard(
                    str(row.get("question", "") or ""),
                    lang=self.article_language,
                ),
                (10, 12, 14),
            ),
        ]
        for prompt_tag, prompt, max_calls_seq in prompt_variants:
            for max_calls in max_calls_seq:
                try:
                    responses = state["agent"].ask(
                        prompt,
                        lang=self.article_language,
                        session_id=(
                            f"{self.setting_name}_{self.article_name}_{row['question_id']}_"
                            f"{run_index}_{prompt_tag}_{max_calls}"
                        ),
                        max_llm_calls_per_run=max_calls,
                        require_tool_use=True,
                        _router_branch_index=run_index,
                    )
                    final_answer = state["agent"].extract_final_text(responses)
                    raw_agent_answer = final_answer
                    tool_uses = state["agent"].extract_tool_uses(responses)
                    strategy_context = state["agent"].get_last_strategy_context()
                    error_text = ""
                    if not tool_uses or _has_error_payload(final_answer):
                        error_text = "missing_tool_use" if not tool_uses else "llm_error_payload"
                        continue
                    break
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    if "maximum context length" not in error_text.lower():
                        break
            if tool_uses and not _has_error_payload(final_answer):
                break

        if tool_uses and (
            not str(final_answer or "").strip()
            or _looks_like_tool_call_payload(final_answer)
            or _has_error_payload(final_answer)
        ):
            final_answer = self._repair_open_answer(
                state=state,
                row=row,
                tool_uses=tool_uses,
                current_answer=final_answer,
            )

        if tool_uses and final_answer:
            final_answer = _ensure_open_answer_payload(final_answer)

        predicted_answer = _extract_semantic_answer_text(final_answer)
        judge_result = {
            "is_correct": False,
            "score": 0.0,
            "reason": error_text or "evaluation_not_run",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
        if tool_uses and predicted_answer:
            try:
                judge_result = state["judge"].evaluate(
                    question=str(row.get("question", "") or ""),
                    reference_answer=str(row.get("reference_answer", "") or ""),
                    candidate_answer=predicted_answer,
                )
            except Exception as exc:
                judge_result = {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": f"judge_error: {type(exc).__name__}: {exc}",
                    "matched_points": [],
                    "missing_points": [],
                    "hallucination_points": [],
                }

        return StageEvalResult(
            setting=self.setting_name,
            article_name=self.article_name,
            run_index=run_index,
            question_id=str(row.get("question_id", "") or ""),
            question=str(row.get("question", "") or ""),
            reference_answer=str(row.get("reference_answer", "") or ""),
            predicted_answer=predicted_answer,
            is_correct=bool(judge_result.get("is_correct", False)),
            latency_ms=int((time.time() - started) * 1000),
            extra={
                "language": self.article_language,
                "tool_call_count": len(tool_uses),
                "tool_names": [
                    str((item or {}).get("tool_name", "") or "").strip()
                    for item in tool_uses
                    if str((item or {}).get("tool_name", "") or "").strip()
                ],
                "tool_uses": tool_uses,
                "strategy_context": strategy_context,
                "raw_agent_answer": raw_agent_answer,
                "judge": judge_result,
                "reference_answers": list(row.get("reference_answers") or []),
                "question_metadata": dict(row.get("metadata") or {}),
                "question_type": str(row.get("question_type", "") or "").strip(),
                "question_type_tags": list(row.get("question_type_tags") or []),
                "related_scenes": list(row.get("related_scenes") or []),
                "evidence_or_reason": str(row.get("evidence_or_reason", "") or ""),
                "error": error_text,
                "required_retrieval_enforced": True,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


def _evaluate_setting(
    *,
    rows: List[Dict[str, Any]],
    evaluator: OpenAgentEvaluator,
    repeats: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    tasks: List[Tuple[Dict[str, Any], int]] = []
    for row in rows:
        for run_index in range(repeats):
            tasks.append((row, run_index))
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers or 1))) as executor:
        future_map = {executor.submit(evaluator.evaluate_row, row, run_index): (row, run_index) for row, run_index in tasks}
        for future in as_completed(future_map):
            results.append(future.result().to_dict())
    results.sort(key=lambda item: (item["article_name"], item["question_id"], int(item["run_index"])))
    return results


def _summarize_stage_setting_base(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_setting(results, question_count=question_count, repeats=repeats)
    scores = [float(((item.get("judge") or {}).get("score", 0.0) or 0.0)) for item in results]
    summary["avg_judge_score"] = round(sum(scores) / float(len(scores) or 1), 4)
    return summary


def _summarize_stage_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_stage_setting_base(results, question_count=question_count, repeats=repeats)

    type_buckets: Dict[str, List[Dict[str, Any]]] = {}
    language_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        question_type = str(item.get("question_type", "") or "").strip()
        if question_type:
            type_buckets.setdefault(question_type, []).append(item)
        language = str(item.get("language", "") or "").strip().lower()
        if language:
            language_buckets.setdefault(language, []).append(item)

    if type_buckets:
        question_type_breakdown: Dict[str, Any] = {}
        for label, bucket in sorted(type_buckets.items()):
            bucket_question_count = len(
                {
                    (str(x.get("article_name", "") or ""), str(x.get("question_id", "") or ""))
                    for x in bucket
                }
            )
            question_type_breakdown[label] = _summarize_stage_setting_base(
                bucket,
                question_count=bucket_question_count,
                repeats=repeats,
            )
        summary["question_type_breakdown"] = question_type_breakdown

    if language_buckets:
        language_breakdown: Dict[str, Any] = {}
        for label, bucket in sorted(language_buckets.items()):
            bucket_question_count = len(
                {
                    (str(x.get("article_name", "") or ""), str(x.get("question_id", "") or ""))
                    for x in bucket
                }
            )
            language_breakdown[label] = _summarize_stage_setting_base(
                bucket,
                question_count=bucket_question_count,
                repeats=repeats,
            )
        summary["language_breakdown"] = language_breakdown

    return summary


def _parse_languages_arg(raw_value: str) -> List[str]:
    text = str(raw_value or "").strip().lower()
    if not text or text == "all":
        return ["zh", "en"]
    values = [piece.strip().lower() for piece in text.replace(",", " ").split() if piece.strip()]
    out: List[str] = []
    for value in values:
        if value in {"zh", "cn", "chinese"} and "zh" not in out:
            out.append("zh")
        if value in {"en", "english"} and "en" not in out:
            out.append("en")
    return out or ["zh", "en"]


def _language_dir_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def _collect_stage_movie_pairs(stage_root: Path, *, languages: Sequence[str]) -> List[Tuple[str, str, Path, Path]]:
    pairs: List[Tuple[str, str, Path, Path]] = []
    for language in languages:
        lang_dir = stage_root / _language_dir_name(language)
        if not lang_dir.exists():
            continue
        for movie_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            script_path = movie_dir / "script.json"
            qa_path = movie_dir / "task_2_question_answering.csv"
            if not script_path.exists() or not qa_path.exists():
                continue
            pairs.append((movie_dir.name, language, script_path, qa_path))
    return pairs


def _load_manifest_movie_pairs(manifest_path: Path, *, languages: Sequence[str]) -> List[Tuple[str, str, Path, Path]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"manifest must be a list: {manifest_path}")
    allowed_languages = set(languages or [])
    pairs: List[Tuple[str, str, Path, Path]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        movie_id = str(item.get("movie_id", "") or "").strip()
        language = str(item.get("language", "") or "").strip().lower()
        script_path = _resolve_cli_path(str(item.get("script_json", "") or ""))
        qa_path = _resolve_cli_path(str(item.get("question_csv", "") or ""))
        if not movie_id or not language or language not in allowed_languages:
            continue
        if not script_path.exists() or not qa_path.exists():
            logger.warning(
                "[STAGE][Manifest] skip movie=%s missing script/question path script=%s question=%s",
                movie_id,
                script_path,
                qa_path,
            )
            continue
        pairs.append((movie_id, language, script_path, qa_path))
    return pairs


def _apply_language_limits(
    movie_pairs: List[Tuple[str, str, Path, Path]],
    *,
    limit_zh_movies: int,
    limit_en_movies: int,
    limit_movies: int,
) -> List[Tuple[str, str, Path, Path]]:
    zh_pairs = [item for item in movie_pairs if item[1] == "zh"]
    en_pairs = [item for item in movie_pairs if item[1] == "en"]

    if limit_zh_movies > 0:
        zh_pairs = zh_pairs[:limit_zh_movies]
    if limit_en_movies > 0:
        en_pairs = en_pairs[:limit_en_movies]

    merged = zh_pairs + en_pairs
    if limit_movies > 0:
        merged = merged[:limit_movies]
    return merged


def _build_base_cfg(config_path: Path) -> KAGConfig:
    cfg = KAGConfig.from_yaml(str(config_path))
    cfg.global_.doc_type = "screenplay"
    cfg.global_config.doc_type = "screenplay"
    cfg.global_.aggregation_mode = "narrative"
    cfg.global_config.aggregation_mode = "narrative"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openai_quality_stable.yaml")
    parser.add_argument("--benchmark-root", default="/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--retrieval-profile", default=DEFAULT_RETRIEVAL_PROFILE)
    parser.add_argument("--workspace-build-mode", default="auto")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--limit-movies", type=int, default=0)
    parser.add_argument("--limit-zh-movies", type=int, default=0)
    parser.add_argument("--limit-en-movies", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-max-workers", type=int, default=8)
    parser.add_argument("--build-max-workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--sentence-chunk-size", type=int, default=None)
    parser.add_argument("--sentence-chunk-overlap", type=int, default=None)
    parser.add_argument("--bm25-chunk-size", type=int, default=None)
    parser.add_argument("--enable-sql-tools", action="store_true", default=True)
    parser.add_argument("--disable-sql-tools", action="store_true")
    parser.add_argument("--rebuild-workspaces", action="store_true")
    parser.add_argument("--workspace-source-root", default="")
    parser.add_argument("--workspace-asset-root", default="experiments/stage/assets/article_workspaces")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-questions-per-movie", type=int, default=0)
    parser.add_argument("--skip-existing-reports", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    benchmark_root = _resolve_cli_path(args.benchmark_root)
    config_path = _resolve_cli_path(args.config)
    manifest_path = _resolve_cli_path(args.manifest_path) if str(args.manifest_path or "").strip() else None
    retrieval_profile = _normalize_retrieval_profile(args.retrieval_profile)
    workspace_build_mode = _resolve_workspace_build_mode(args.workspace_build_mode, retrieval_profile=retrieval_profile)
    setting_name = _setting_name_for_profile(retrieval_profile)
    selected_languages = _parse_languages_arg(args.languages)
    workspace_source_root = _resolve_cli_path(args.workspace_source_root) if str(args.workspace_source_root or "").strip() else None
    workspace_asset_root = _resolve_cli_path(args.workspace_asset_root) if str(args.workspace_asset_root or "").strip() else _default_workspace_asset_root()
    workspace_asset_root.mkdir(parents=True, exist_ok=True)
    converted_script_root = _default_converted_script_root()
    converted_script_root.mkdir(parents=True, exist_ok=True)

    run_name = str(args.run_name or "").strip() or time.strftime("stage_task2_no_strategy_%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "experiments" / "stage" / "runs" / run_name
    report_root = run_root / "reports"
    workspace_root = run_root / "article_workspaces"
    for path in (run_root, report_root, workspace_root):
        path.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_cfg(config_path)
    enable_sql_tools = _profile_uses_sql_tools(
        retrieval_profile,
        cli_enable_sql_tools=bool(args.enable_sql_tools) and not bool(args.disable_sql_tools),
    )

    if manifest_path is not None:
        movie_pairs = _load_manifest_movie_pairs(manifest_path, languages=selected_languages)
        movie_pairs = _apply_language_limits(
            movie_pairs,
            limit_zh_movies=max(0, int(args.limit_zh_movies or 0)),
            limit_en_movies=max(0, int(args.limit_en_movies or 0)),
            limit_movies=max(0, int(args.limit_movies or 0)),
        )
    else:
        movie_pairs = _collect_stage_movie_pairs(benchmark_root, languages=selected_languages)
        movie_pairs = _apply_language_limits(
            movie_pairs,
            limit_zh_movies=max(0, int(args.limit_zh_movies or 0)),
            limit_en_movies=max(0, int(args.limit_en_movies or 0)),
            limit_movies=max(0, int(args.limit_movies or 0)),
        )
    manifest = [
        {
            "movie_id": movie_id,
            "language": language,
            "script_json": str(script_path),
            "question_csv": str(question_path),
        }
        for movie_id, language, script_path, question_path in movie_pairs
    ]
    json_dump_atomic(str(run_root / "manifest.json"), manifest)
    experiment_lines = [
        "# STAGE Task 2 Experiment",
        "",
        f"- run_name: {run_name}",
        f"- retrieval_profile: {retrieval_profile}",
        f"- workspace_build_mode: {workspace_build_mode}",
        f"- config: {config_path}",
        f"- benchmark_root: {benchmark_root}",
        f"- manifest_path: {manifest_path if manifest_path is not None else (run_root / 'manifest.json')}",
        f"- workspace_asset_root: {workspace_asset_root}",
        f"- workspace_source_root: {workspace_source_root or ''}",
        f"- build_max_workers: {max(1, int(args.build_max_workers or 32))}",
        f"- eval_max_workers: {max(1, int(args.eval_max_workers or 1))}",
        f"- repeats: {max(1, int(args.repeats or 1))}",
        f"- chunk_size: {args.chunk_size if args.chunk_size is not None else 'config_default'}",
        f"- chunk_overlap: {args.chunk_overlap if args.chunk_overlap is not None else 'config_default'}",
        f"- sentence_chunk_size: {args.sentence_chunk_size if args.sentence_chunk_size is not None else 'auto_or_config_default'}",
        f"- sentence_chunk_overlap: {args.sentence_chunk_overlap if args.sentence_chunk_overlap is not None else 'auto_or_config_default'}",
        f"- bm25_chunk_size: {args.bm25_chunk_size if args.bm25_chunk_size is not None else 'auto_or_config_default'}",
        f"- enable_sql_tools: {enable_sql_tools}",
        f"- rebuild_workspaces: {bool(args.rebuild_workspaces)}",
        f"- skip_existing_reports: {bool(args.skip_existing_reports)}",
        "",
        "## Selected Movies",
        "",
    ]
    for movie_id, language, script_path, question_path in movie_pairs:
        experiment_lines.append(f"- {movie_id} ({language}): script={script_path} qa={question_path}")
    (run_root / "experiment.md").write_text("\n".join(experiment_lines) + "\n", encoding="utf-8")

    all_results: List[Dict[str, Any]] = []
    movie_summaries: Dict[str, Any] = {}
    total_questions = 0
    progress_path = report_root / "progress.json"

    for movie_index, (movie_id, language, script_src_path, question_csv_path) in enumerate(movie_pairs, start=1):
        report_path = report_root / f"{movie_id}.json"
        if bool(args.skip_existing_reports) and report_path.exists():
            try:
                existing_payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[STAGE][Skip] failed to load existing report for movie=%s err=%s", movie_id, e)
                existing_payload = None
            if isinstance(existing_payload, dict):
                existing_results = existing_payload.get("results")
                existing_summary = existing_payload.get("summary")
                if isinstance(existing_results, list) and isinstance(existing_summary, dict):
                    question_count = int(existing_payload.get("question_count") or existing_summary.get("question_count") or 0)
                    total_questions += question_count
                    movie_summaries[movie_id] = {
                        "language": language,
                        **existing_summary,
                    }
                    all_results.extend([item for item in existing_results if isinstance(item, dict)])
                    _write_setting_progress(
                        path=progress_path,
                        setting_name=setting_name,
                        article_name=movie_id,
                        repeats=max(1, int(args.repeats or 1)),
                        repeat_index=max(1, int(args.repeats or 1)),
                        batch_index=movie_index,
                        batch_total=len(movie_pairs),
                        phase="completed_movie",
                        evaluated_attempts_done=len(all_results),
                        evaluated_attempts_total=total_questions * max(1, int(args.repeats or 1)),
                        batch_question_count=question_count,
                        note=f"skipped existing report {movie_index}/{len(movie_pairs)} movies",
                    )
                    logger.info(
                        "[%d/%d] movie=%s lang=%s skipped existing report overall=%.4f pass=%.4f avg_judge=%.4f",
                        movie_index,
                        len(movie_pairs),
                        movie_id,
                        language,
                        float(existing_summary.get("overall_accuracy", 0.0) or 0.0),
                        float(existing_summary.get("pass_accuracy", 0.0) or 0.0),
                        float(existing_summary.get("avg_judge_score", 0.0) or 0.0),
                    )
                    continue

        converted_script_path = converted_script_root / f"{movie_id}.json"
        if not converted_script_path.exists():
            _convert_stage_script(
                movie_id=movie_id,
                language=language,
                src_path=script_src_path,
                dst_path=converted_script_path,
            )

        workspace_dir = _resolve_article_workspace_dir(
            article_name=movie_id,
            workspace_root=workspace_root,
            workspace_source_root=workspace_source_root,
            workspace_asset_root=workspace_asset_root,
        )
        cfg = _prepare_stage_article_cfg(
            base_cfg=base_cfg,
            article_name=movie_id,
            article_json_path=converted_script_path,
            workspace_dir=workspace_dir,
            workspace_asset_root=workspace_asset_root,
            language=language,
            build_max_workers=max(1, int(args.build_max_workers or 32)),
            retrieval_profile=retrieval_profile,
            workspace_build_mode=workspace_build_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            sentence_chunk_size=args.sentence_chunk_size,
            sentence_chunk_overlap=args.sentence_chunk_overlap,
            bm25_chunk_size=args.bm25_chunk_size,
            rebuild=bool(args.rebuild_workspaces),
        )
        cfg.strategy_memory.hidden_tool_names = list({
            *list(getattr(cfg.strategy_memory, "hidden_tool_names", []) or []),
            "choice_grounded_evidence_search",
        })

        qa_rows = _load_stage_task2_qas(question_csv_path)
        if args.max_questions_per_movie and args.max_questions_per_movie > 0:
            qa_rows = qa_rows[: int(args.max_questions_per_movie)]
        total_questions += len(qa_rows)

        evaluator = OpenAgentEvaluator(
            cfg,
            setting_name=setting_name,
            retrieval_profile=retrieval_profile,
            article_name=movie_id,
            article_language=language,
            enable_sql_tools=enable_sql_tools,
        )
        try:
            movie_results = _evaluate_setting(
                rows=qa_rows,
                evaluator=evaluator,
                repeats=max(1, int(args.repeats or 1)),
                max_workers=max(1, int(args.eval_max_workers or 1)),
            )
        finally:
            evaluator.close()

        summary = _summarize_stage_setting(
            movie_results,
            question_count=len(qa_rows),
            repeats=max(1, int(args.repeats or 1)),
        )
        movie_payload = {
            "setting": setting_name,
            "movie_id": movie_id,
            "language": language,
            "retrieval_profile": retrieval_profile,
            "workspace_build_mode": workspace_build_mode,
            "question_count": len(qa_rows),
            "repeats": max(1, int(args.repeats or 1)),
            "results": movie_results,
            "summary": summary,
        }
        json_dump_atomic(str(report_path), movie_payload)
        movie_summaries[movie_id] = {
            "language": language,
            **summary,
        }
        all_results.extend(movie_results)

        _write_setting_progress(
            path=progress_path,
            setting_name=setting_name,
            article_name=movie_id,
            repeats=max(1, int(args.repeats or 1)),
            repeat_index=max(1, int(args.repeats or 1)),
            batch_index=movie_index,
            batch_total=len(movie_pairs),
            phase="completed_movie",
            evaluated_attempts_done=len(all_results),
            evaluated_attempts_total=total_questions * max(1, int(args.repeats or 1)),
            batch_question_count=len(qa_rows),
            note=f"completed {movie_index}/{len(movie_pairs)} movies",
        )
        logger.info(
            "[%d/%d] movie=%s lang=%s overall=%.4f pass=%.4f avg_judge=%.4f",
            movie_index,
            len(movie_pairs),
            movie_id,
            language,
            float(summary.get("overall_accuracy", 0.0) or 0.0),
            float(summary.get("pass_accuracy", 0.0) or 0.0),
            float(summary.get("avg_judge_score", 0.0) or 0.0),
        )

    overall_summary = _summarize_stage_setting(
        all_results,
        question_count=total_questions,
        repeats=max(1, int(args.repeats or 1)),
    )
    final_payload = {
        "benchmark_root": str(benchmark_root),
        "config": str(config_path),
        "setting": setting_name,
        "retrieval_profile": retrieval_profile,
        "workspace_build_mode": workspace_build_mode,
        "movie_count": len(movie_pairs),
        "question_count": total_questions,
        "repeats": max(1, int(args.repeats or 1)),
        "build_max_workers": max(1, int(args.build_max_workers or 32)),
        "eval_max_workers": max(1, int(args.eval_max_workers or 1)),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "sentence_chunk_size": args.sentence_chunk_size,
        "sentence_chunk_overlap": args.sentence_chunk_overlap,
        "bm25_chunk_size": args.bm25_chunk_size,
        "enable_sql_tools": enable_sql_tools,
        "summary": overall_summary,
        "movie_summaries": movie_summaries,
        "selected_movies": [
            {
                "movie_id": movie_id,
                "language": language,
                "script_json": str(script_path),
                "question_csv": str(question_path),
            }
            for movie_id, language, script_path, question_path in movie_pairs
        ],
        "manifest_path": str(run_root / "manifest.json"),
    }
    json_dump_atomic(str(report_root / "summary.json"), final_payload)

    md_lines = [
        f"# STAGE Task 2 {setting_name} Summary",
        "",
        f"- retrieval_profile: {retrieval_profile}",
        f"- workspace_build_mode: {workspace_build_mode}",
        f"- movie_count: {len(movie_pairs)}",
        f"- question_count: {total_questions}",
        f"- repeats: {max(1, int(args.repeats or 1))}",
        f"- build_max_workers: {max(1, int(args.build_max_workers or 32))}",
        f"- eval_max_workers: {max(1, int(args.eval_max_workers or 1))}",
        f"- chunk_size: {args.chunk_size if args.chunk_size is not None else 'config_default'}",
        f"- chunk_overlap: {args.chunk_overlap if args.chunk_overlap is not None else 'config_default'}",
        f"- sentence_chunk_size: {args.sentence_chunk_size if args.sentence_chunk_size is not None else 'auto_or_config_default'}",
        f"- sentence_chunk_overlap: {args.sentence_chunk_overlap if args.sentence_chunk_overlap is not None else 'auto_or_config_default'}",
        f"- bm25_chunk_size: {args.bm25_chunk_size if args.bm25_chunk_size is not None else 'auto_or_config_default'}",
        f"- enable_sql_tools: {enable_sql_tools}",
        f"- languages: {', '.join(selected_languages)}",
        f"- limit_zh_movies: {max(0, int(args.limit_zh_movies or 0))}",
        f"- limit_en_movies: {max(0, int(args.limit_en_movies or 0))}",
        f"- limit_movies: {max(0, int(args.limit_movies or 0))}",
        f"- benchmark_root: {benchmark_root}",
        f"- config: {config_path}",
        f"- overall_accuracy: {overall_summary['overall_accuracy']}",
        f"- pass_accuracy: {overall_summary['pass_accuracy']}",
        f"- avg_judge_score: {overall_summary['avg_judge_score']}",
        f"- avg_latency_ms: {overall_summary['avg_latency_ms']}",
        "",
        "## Selected Movies",
        "",
    ]
    for movie_id, language, script_path, question_path in movie_pairs:
        md_lines.append(f"- {movie_id} ({language}): script={script_path} qa={question_path}")

    md_lines.extend(
        [
            "",
        "## Per Movie",
        "",
        ]
    )
    for movie_id, summary in sorted(movie_summaries.items()):
        md_lines.append(
            f"- {movie_id} ({summary.get('language', '')}): overall={summary['overall_accuracy']} pass={summary['pass_accuracy']} avg_judge={summary['avg_judge_score']}"
        )

    language_breakdown = overall_summary.get("language_breakdown") if isinstance(overall_summary.get("language_breakdown"), dict) else {}
    if language_breakdown:
        md_lines.extend(["", "## Language Breakdown", ""])
        for label, bucket_summary in sorted(language_breakdown.items()):
            md_lines.append(
                f"- {label}: overall={bucket_summary['overall_accuracy']} pass={bucket_summary['pass_accuracy']} avg_judge={bucket_summary['avg_judge_score']}"
            )

    question_type_breakdown = overall_summary.get("question_type_breakdown") if isinstance(overall_summary.get("question_type_breakdown"), dict) else {}
    if question_type_breakdown:
        md_lines.extend(["", "## Question Type Breakdown", ""])
        for label, bucket_summary in sorted(question_type_breakdown.items()):
            md_lines.append(
                f"- {label}: overall={bucket_summary['overall_accuracy']} pass={bucket_summary['pass_accuracy']} avg_judge={bucket_summary['avg_judge_score']}"
            )

    (report_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Finished STAGE task_2 benchmark. summary=%s", report_root / "summary.json")


if __name__ == "__main__":
    main()
