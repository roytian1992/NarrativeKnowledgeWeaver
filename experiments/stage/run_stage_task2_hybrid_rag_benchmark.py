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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.model_providers.openai_llm import OpenAILLM
from core.models.data import Document as KAGDocument
from core.storage.vector_store import VectorStore
from core.utils.config import KAGConfig, _apply_global_locale_paths
from core.utils.general_utils import (
    compress_query_for_vector_search,
    json_dump_atomic,
    parse_json_object_from_text,
    word_len,
)
from core.utils.prompt_loader import YAMLPromptLoader
from retriever.sparse_retriever import KeywordBM25Retriever

logger = logging.getLogger(__name__)


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


def _resolve_cli_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _close_vector_store(store: Any) -> None:
    closer = getattr(store, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _default_workspace_asset_root() -> Path:
    return REPO_ROOT / "experiments" / "stage" / "assets" / "article_workspaces_hybridrag_plain"


def _default_converted_script_root() -> Path:
    return REPO_ROOT / "experiments" / "stage" / "assets" / "converted_scripts"


def _set_global_language(cfg: KAGConfig, language: str) -> None:
    cfg.global_.language = language
    cfg.global_.locale = language
    cfg.global_config.language = language
    cfg.global_config.locale = language
    _apply_global_locale_paths(cfg.global_)
    _apply_global_locale_paths(cfg.global_config)


def _build_stage_signature(
    cfg: KAGConfig,
    *,
    retrieval_mode: str,
    fusion_mode: str,
) -> Dict[str, Any]:
    return {
        "retrieval_mode": str(retrieval_mode or "hybrid_rag_plain"),
        "fusion_mode": str(fusion_mode or "rrf"),
        "chunk_size": int(getattr(cfg.document_processing, "chunk_size", 0) or 0),
        "chunk_overlap": int(getattr(cfg.document_processing, "chunk_overlap", 0) or 0),
        "dense_top_k": int(getattr(cfg.document_processing, "dense_top_k", 0) or 0),
        "bm25_top_k": int(getattr(cfg.document_processing, "bm25_top_k", 0) or 0),
        "final_top_k": int(getattr(cfg.document_processing, "final_top_k", 0) or 0),
    }


def _read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _marker_matches(marker_payload: Dict[str, Any], *, signature: Dict[str, Any]) -> bool:
    if not isinstance(marker_payload, dict) or not marker_payload:
        return False
    for key, expected in signature.items():
        if marker_payload.get(key) != expected:
            return False
    return True


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
            "source_scene_subtitle": subtitle,
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
    return cfg


def _build_movie_cfg(
    base_cfg: KAGConfig,
    *,
    workspace_dir: Path,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
    dense_top_k: int,
    bm25_top_k: int,
    final_top_k: int,
) -> KAGConfig:
    cfg = copy.deepcopy(base_cfg)
    _set_global_language(cfg, language)
    cfg.global_.doc_type = "screenplay"
    cfg.global_config.doc_type = "screenplay"
    cfg.document_processing.chunk_size = int(chunk_size)
    cfg.document_processing.chunk_overlap = int(chunk_overlap)
    cfg.document_processing.dense_top_k = int(dense_top_k)
    cfg.document_processing.bm25_top_k = int(bm25_top_k)
    cfg.document_processing.final_top_k = int(final_top_k)
    cfg.storage.vector_store_path = str(workspace_dir / "vector_store")
    return cfg


def _chunk_scene_documents(
    *,
    movie_id: str,
    language: str,
    converted_script_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    scenes = json.loads(converted_script_path.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise ValueError(f"Expected converted script list: {converted_script_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(10, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
        length_function=lambda text: word_len(text, lang="auto"),
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？", "；", "：",
            ". ", "? ", "! ", "; ", ": ",
            "，", ", ", " ",
            "",
        ],
        keep_separator=True,
    )

    chunks: List[Dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        content = str(scene.get("content", "") or "").strip()
        if not content:
            continue
        scene_id = str(scene.get("id", "") or f"{movie_id}_scene_{scene_index}")
        title = str(scene.get("title", "") or "").strip()
        subtitle = str(scene.get("subtitle", "") or "").strip()
        scene_header = "\n".join(part for part in [title, subtitle] if part).strip()
        payload_text = f"{scene_header}\n{content}".strip() if scene_header else content
        parts = splitter.split_text(payload_text) if word_len(payload_text, lang="auto") > int(chunk_size) else [payload_text]
        for chunk_index, chunk_text in enumerate(parts, start=1):
            text = str(chunk_text or "").strip()
            if not text:
                continue
            chunk_id = f"{scene_id}_chunk_{chunk_index:03d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "content": text,
                    "metadata": {
                        "movie_id": movie_id,
                        "language": language,
                        "scene_id": scene_id,
                        "scene_order": scene_index,
                        "title": title,
                        "subtitle": subtitle,
                        "chunk_order": chunk_index,
                        "document_id": scene_id,
                        "chunk_id": chunk_id,
                    },
                }
            )
    return chunks


def _zh_preprocess(text: str) -> List[str]:
    try:
        import re
        import jieba

        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return list(jieba.cut(text, cut_all=False))
    except Exception:
        return (text or "").split()


def _build_workspace_assets(
    *,
    cfg: KAGConfig,
    workspace_dir: Path,
    converted_script_path: Path,
    movie_id: str,
    language: str,
    retrieval_mode: str,
    fusion_mode: str,
    rebuild: bool,
) -> Dict[str, Any]:
    signature = _build_stage_signature(cfg, retrieval_mode=retrieval_mode, fusion_mode=fusion_mode)
    marker_path = workspace_dir / "build_marker.json"
    chunks_path = workspace_dir / "chunks.json"
    marker_payload = _read_json_dict(marker_path)

    if (
        not rebuild
        and marker_path.exists()
        and chunks_path.exists()
        and _marker_matches(marker_payload, signature=signature)
    ):
        return marker_payload

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_scene_documents(
        movie_id=movie_id,
        language=language,
        converted_script_path=converted_script_path,
        chunk_size=int(cfg.document_processing.chunk_size),
        chunk_overlap=int(cfg.document_processing.chunk_overlap),
    )
    json_dump_atomic(str(chunks_path), chunks)

    vector_store = VectorStore(cfg, category="chunks")
    try:
        vector_store.delete_collection()
        vector_store.store_documents(
            [
                KAGDocument(id=str(item["id"]), content=str(item["content"]), metadata=dict(item.get("metadata") or {}))
                for item in chunks
            ]
        )
    finally:
        _close_vector_store(vector_store)

    marker = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "movie_id": movie_id,
        "language": language,
        "converted_script_path": str(converted_script_path),
        "workspace_dir": str(workspace_dir),
        "chunks_path": str(chunks_path),
        "chunk_count": len(chunks),
        **signature,
    }
    json_dump_atomic(str(marker_path), marker)
    return marker


def _load_workspace_chunks(workspace_dir: Path) -> List[Dict[str, Any]]:
    chunks_path = workspace_dir / "chunks.json"
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid chunks payload: {chunks_path}")
    return [item for item in payload if isinstance(item, dict)]


def _format_evidence_chunks(chunks: List[Dict[str, Any]], *, limit_chars: int = 7000) -> str:
    blocks: List[str] = []
    used = 0
    for index, item in enumerate(chunks, start=1):
        md = dict(item.get("metadata") or {})
        title = str(md.get("title", "") or "").strip()
        subtitle = str(md.get("subtitle", "") or "").strip()
        scene_id = str(md.get("scene_id", "") or "").strip()
        chunk_id = str(item.get("id", "") or "").strip()
        header_parts = [part for part in [title, subtitle] if part]
        header = " / ".join(header_parts) if header_parts else scene_id or chunk_id
        block = "\n".join(
            [
                f"[{index}] {header}",
                f"chunk_id: {chunk_id}",
                str(item.get("content", "") or "").strip(),
            ]
        ).strip()
        if not block:
            continue
        projected = used + len(block) + (4 if blocks else 0)
        if blocks and projected > limit_chars:
            break
        blocks.append(block)
        used = projected
    return "\n\n---\n\n".join(blocks)


def _extract_answer_text(answer_text: str) -> str:
    payload = parse_json_object_from_text(answer_text)
    if isinstance(payload, dict):
        for key in ("answer_text", "answer", "final_answer"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
    return str(answer_text or "").strip()


def _ensure_answer_payload(answer_text: str, *, default_confidence: float = 0.7) -> str:
    payload = parse_json_object_from_text(answer_text)
    if isinstance(payload, dict):
        answer_value = ""
        for key in ("answer_text", "answer", "final_answer"):
            answer_value = str(payload.get(key, "") or "").strip()
            if answer_value:
                break
        if answer_value:
            payload["answer_text"] = answer_value
            if payload.get("evidence") is None:
                payload["evidence"] = ""
            if payload.get("confidence") is None:
                payload["confidence"] = float(default_confidence)
            return json.dumps(payload, ensure_ascii=False)
    semantic = _extract_answer_text(answer_text)
    return json.dumps(
        {
            "answer_text": semantic,
            "evidence": "",
            "confidence": float(default_confidence),
        },
        ensure_ascii=False,
    )


def _build_answer_prompt(
    *,
    question: str,
    evidence_text: str,
    language: str,
) -> str:
    if language == "zh":
        return "\n".join(
            [
                "你是一个基于证据回答问题的助手。",
                "你只能根据给定证据回答，不要使用外部知识。",
                "如果证据不足，请保守回答，不要编造。",
                "",
                "只返回 JSON：",
                '{"answer_text":"...","evidence":"...","confidence":0.72}',
                "",
                "规则：",
                "- `answer_text` 必须直接回答问题，用一句简短的话或短语。",
                "- `evidence` 必须简短概括你依赖的证据。",
                "- `confidence` 必须是 0 到 1 之间的数字。",
                "- 不要输出推理过程或额外解释。",
                "",
                "问题：",
                str(question or "").strip(),
                "",
                "证据：",
                evidence_text or "(无证据)",
            ]
        ).strip()
    return "\n".join(
        [
            "You are a retrieval-grounded QA assistant.",
            "Answer only from the provided evidence and do not use outside knowledge.",
            "If the evidence is insufficient, answer conservatively instead of inventing details.",
            "",
            "Return JSON only:",
            '{"answer_text":"...","evidence":"...","confidence":0.72}',
            "",
            "Rules:",
            "- `answer_text` must directly answer the question in one short sentence or phrase.",
            "- `evidence` must briefly summarize the supporting evidence.",
            "- `confidence` must be a number between 0 and 1.",
            "- Do not output chain-of-thought or extra explanation.",
            "",
            "Question:",
            str(question or "").strip(),
            "",
            "Evidence:",
            evidence_text or "(no evidence)",
        ]
    ).strip()


def _rrf_fuse(
    *,
    dense_hits: List[Dict[str, Any]],
    sparse_hits: List[Dict[str, Any]],
    final_top_k: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}

    def _touch(hit: Dict[str, Any], source: str, rank: int) -> None:
        chunk_id = str(hit.get("id", "") or "").strip()
        if not chunk_id:
            return
        row = fused.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "content": str(hit.get("content", "") or ""),
                "metadata": dict(hit.get("metadata") or {}),
                "dense_rank": None,
                "dense_similarity": None,
                "sparse_rank": None,
                "rrf_score": 0.0,
                "sources": [],
            },
        )
        row["rrf_score"] += 1.0 / float(rrf_k + rank)
        if source == "dense":
            row["dense_rank"] = rank
            row["dense_similarity"] = float(hit.get("dense_similarity", 0.0) or 0.0)
        if source == "sparse":
            row["sparse_rank"] = rank
        if source not in row["sources"]:
            row["sources"].append(source)

    for rank, hit in enumerate(dense_hits, start=1):
        _touch(hit, "dense", rank)
    for rank, hit in enumerate(sparse_hits, start=1):
        _touch(hit, "sparse", rank)

    rows = list(fused.values())
    rows.sort(
        key=lambda item: (
            float(item.get("rrf_score", 0.0) or 0.0),
            len(item.get("sources", []) or []),
            float(item.get("dense_similarity", 0.0) or 0.0),
            -int(item.get("sparse_rank", 10**9) or 10**9),
        ),
        reverse=True,
    )
    return rows[: max(1, int(final_top_k or 1))]


class HybridRAGMovieRuntime:
    def __init__(self, cfg: KAGConfig, *, workspace_dir: Path) -> None:
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.chunks = _load_workspace_chunks(workspace_dir)
        self.chunk_map = {str(item.get("id", "") or ""): item for item in self.chunks}
        self.vector_store = VectorStore(cfg, category="chunks")
        self.bm25 = KeywordBM25Retriever(
            [
                LCDocument(page_content=str(item.get("content", "") or ""), metadata=dict(item.get("metadata") or {}) | {"chunk_id": str(item.get("id", "") or "")})
                for item in self.chunks
            ],
            zh_preprocess=_zh_preprocess,
            k_default=max(1, int(getattr(cfg.document_processing, "bm25_top_k", 8) or 8)),
        )

    def retrieve(self, question: str) -> Dict[str, Any]:
        dense_top_k = max(1, int(getattr(self.cfg.document_processing, "dense_top_k", 8) or 8))
        bm25_top_k = max(1, int(getattr(self.cfg.document_processing, "bm25_top_k", 8) or 8))
        final_top_k = max(1, int(getattr(self.cfg.document_processing, "final_top_k", 8) or 8))

        dense_query = compress_query_for_vector_search(question, top_k=8) or str(question or "").strip()
        dense_results = self.vector_store.search(query=dense_query, limit=dense_top_k)
        dense_hits: List[Dict[str, Any]] = []
        for doc in dense_results:
            chunk_id = str(doc.id or (doc.metadata or {}).get("chunk_id") or "").strip()
            base = self.chunk_map.get(chunk_id)
            if base is None:
                continue
            dense_hits.append(
                {
                    "id": chunk_id,
                    "content": str(base.get("content", "") or ""),
                    "metadata": dict(base.get("metadata") or {}),
                    "dense_similarity": float((doc.metadata or {}).get("similarity_score", 0.0) or 0.0),
                }
            )

        sparse_results = self.bm25.retrieve(query=str(question or "").strip(), k=bm25_top_k)
        sparse_hits: List[Dict[str, Any]] = []
        for doc in sparse_results:
            chunk_id = str((doc.metadata or {}).get("chunk_id", "") or "").strip()
            base = self.chunk_map.get(chunk_id)
            if base is None:
                continue
            sparse_hits.append(
                {
                    "id": chunk_id,
                    "content": str(base.get("content", "") or ""),
                    "metadata": dict(base.get("metadata") or {}),
                }
            )

        fused = _rrf_fuse(dense_hits=dense_hits, sparse_hits=sparse_hits, final_top_k=final_top_k)
        return {
            "dense_query": dense_query,
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits,
            "fused_hits": fused,
            "evidence_text": _format_evidence_chunks(fused),
        }

    def close(self) -> None:
        try:
            _close_vector_store(self.vector_store)
        except Exception:
            pass


class HybridRAGThreadLocal:
    def __init__(
        self,
        cfg: KAGConfig,
        *,
        workspace_dir: Path,
        article_language: str,
    ) -> None:
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.article_language = str(article_language or "").strip().lower() or "en"
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _build_state(self) -> Dict[str, Any]:
        runtime = HybridRAGMovieRuntime(self.cfg, workspace_dir=self.workspace_dir)
        prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        answer_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge = RetrievalAnswerJudge(
            prompt_loader=prompt_loader,
            llm=judge_llm,
            prompt_id="memory/judge_open_retrieval_answer",
        )
        state = {
            "runtime": runtime,
            "answer_llm": answer_llm,
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
                state["runtime"].close()
            except Exception:
                pass


class HybridRAGEvaluator:
    def __init__(
        self,
        cfg: KAGConfig,
        *,
        setting_name: str,
        article_name: str,
        workspace_dir: Path,
        article_language: str,
    ) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.article_name = article_name
        self.article_language = str(article_language or "").strip().lower() or "en"
        self.tlocal = HybridRAGThreadLocal(
            cfg,
            workspace_dir=workspace_dir,
            article_language=self.article_language,
        )

    def evaluate_row(self, row: Dict[str, Any], run_index: int) -> StageEvalResult:
        started = time.time()
        state = self.tlocal.state()
        retrieval = state["runtime"].retrieve(str(row.get("question", "") or ""))
        prompt = _build_answer_prompt(
            question=str(row.get("question", "") or ""),
            evidence_text=str(retrieval.get("evidence_text", "") or ""),
            language=self.article_language,
        )
        answer_raw = ""
        judge_result = {
            "is_correct": False,
            "score": 0.0,
            "reason": "evaluation_not_run",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
        error_text = ""
        try:
            result = state["answer_llm"].run([{"role": "user", "content": prompt}])
            answer_raw = str(((result or [{}])[0] or {}).get("content", "") or "").strip()
            answer_raw = _ensure_answer_payload(answer_raw)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            answer_raw = _ensure_answer_payload("")

        predicted_answer = _extract_answer_text(answer_raw)
        if predicted_answer:
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
                "raw_answer": answer_raw,
                "judge": judge_result,
                "reference_answers": list(row.get("reference_answers") or []),
                "question_metadata": dict(row.get("metadata") or {}),
                "question_type": str(row.get("question_type", "") or "").strip(),
                "question_type_tags": list(row.get("question_type_tags") or []),
                "related_scenes": list(row.get("related_scenes") or []),
                "evidence_or_reason": str(row.get("evidence_or_reason", "") or ""),
                "retrieval": retrieval,
                "error": error_text,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


def _evaluate_setting(
    *,
    rows: List[Dict[str, Any]],
    evaluator: HybridRAGEvaluator,
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


def _summarize_setting_base(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    total_attempts = len(results)
    correct_attempts = sum(1 for item in results if bool(item.get("is_correct", False)))
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in results:
        key = (str(item.get("article_name", "") or ""), str(item.get("question_id", "") or ""))
        grouped.setdefault(key, []).append(item)
    pass_question_count = sum(1 for bucket in grouped.values() if any(bool(item.get("is_correct", False)) for item in bucket))
    avg_latency = round(sum(int(item.get("latency_ms", 0) or 0) for item in results) / float(total_attempts or 1), 2)
    avg_judge_score = round(
        sum(float(((item.get("judge") or {}).get("score", 0.0) or 0.0)) for item in results) / float(total_attempts or 1),
        4,
    )
    return {
        "question_count": question_count,
        "repeats": repeats,
        "total_attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "overall_accuracy": round(correct_attempts / float(total_attempts or 1), 4),
        "pass_accuracy": round(pass_question_count / float(question_count or 1), 4),
        "pass_question_count": pass_question_count,
        "avg_latency_ms": avg_latency,
        "avg_judge_score": avg_judge_score,
    }


def _summarize_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_setting_base(results, question_count=question_count, repeats=repeats)
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
        summary["question_type_breakdown"] = {}
        for label, bucket in sorted(type_buckets.items()):
            bucket_question_count = len({(str(x.get("article_name", "") or ""), str(x.get("question_id", "") or "")) for x in bucket})
            summary["question_type_breakdown"][label] = _summarize_setting_base(bucket, question_count=bucket_question_count, repeats=repeats)
    if language_buckets:
        summary["language_breakdown"] = {}
        for label, bucket in sorted(language_buckets.items()):
            bucket_question_count = len({(str(x.get("article_name", "") or ""), str(x.get("question_id", "") or "")) for x in bucket})
            summary["language_breakdown"][label] = _summarize_setting_base(bucket, question_count=bucket_question_count, repeats=repeats)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openai_quality_stable.yaml")
    parser.add_argument("--benchmark-root", default="/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--limit-movies", type=int, default=0)
    parser.add_argument("--limit-zh-movies", type=int, default=0)
    parser.add_argument("--limit-en-movies", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-max-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=8)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--rebuild-workspaces", action="store_true")
    parser.add_argument("--workspace-asset-root", default="experiments/stage/assets/article_workspaces_hybridrag_plain")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-questions-per-movie", type=int, default=0)
    parser.add_argument("--skip-existing-reports", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    benchmark_root = _resolve_cli_path(args.benchmark_root)
    config_path = _resolve_cli_path(args.config)
    manifest_path = _resolve_cli_path(args.manifest_path) if str(args.manifest_path or "").strip() else None
    selected_languages = _parse_languages_arg(args.languages)
    workspace_asset_root = _resolve_cli_path(args.workspace_asset_root) if str(args.workspace_asset_root or "").strip() else _default_workspace_asset_root()
    workspace_asset_root.mkdir(parents=True, exist_ok=True)
    converted_script_root = _default_converted_script_root()
    converted_script_root.mkdir(parents=True, exist_ok=True)

    run_name = str(args.run_name or "").strip() or time.strftime("stage_task2_hybridrag_plain_%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "experiments" / "stage" / "runs" / run_name
    report_root = run_root / "reports"
    for path in (run_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_cfg(config_path)
    setting_name = "hybrid_rag_plain"
    retrieval_mode = "hybrid_rag_plain"
    fusion_mode = "rrf"

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
        "# STAGE Task 2 Hybrid RAG Experiment",
        "",
        f"- run_name: {run_name}",
        f"- setting: {setting_name}",
        f"- retrieval_mode: {retrieval_mode}",
        f"- fusion_mode: {fusion_mode}",
        f"- config: {config_path}",
        f"- benchmark_root: {benchmark_root}",
        f"- manifest_path: {manifest_path if manifest_path is not None else (run_root / 'manifest.json')}",
        f"- workspace_asset_root: {workspace_asset_root}",
        f"- repeats: {max(1, int(args.repeats or 1))}",
        f"- eval_max_workers: {max(1, int(args.eval_max_workers or 1))}",
        f"- chunk_size: {int(args.chunk_size)}",
        f"- chunk_overlap: {int(args.chunk_overlap)}",
        f"- dense_top_k: {int(args.dense_top_k)}",
        f"- bm25_top_k: {int(args.bm25_top_k)}",
        f"- final_top_k: {int(args.final_top_k)}",
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
            except Exception:
                existing_payload = None
            if isinstance(existing_payload, dict):
                existing_results = existing_payload.get("results")
                existing_summary = existing_payload.get("summary")
                if isinstance(existing_results, list) and isinstance(existing_summary, dict):
                    question_count = int(existing_payload.get("question_count") or existing_summary.get("question_count") or 0)
                    total_questions += question_count
                    movie_summaries[movie_id] = {"language": language, **existing_summary}
                    all_results.extend([item for item in existing_results if isinstance(item, dict)])
                    continue

        converted_script_path = converted_script_root / f"{movie_id}.json"
        if not converted_script_path.exists():
            _convert_stage_script(
                movie_id=movie_id,
                language=language,
                src_path=script_src_path,
                dst_path=converted_script_path,
            )

        workspace_dir = workspace_asset_root / movie_id
        cfg = _build_movie_cfg(
            base_cfg,
            workspace_dir=workspace_dir,
            language=language,
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
            dense_top_k=int(args.dense_top_k),
            bm25_top_k=int(args.bm25_top_k),
            final_top_k=int(args.final_top_k),
        )
        _build_workspace_assets(
            cfg=cfg,
            workspace_dir=workspace_dir,
            converted_script_path=converted_script_path,
            movie_id=movie_id,
            language=language,
            retrieval_mode=retrieval_mode,
            fusion_mode=fusion_mode,
            rebuild=bool(args.rebuild_workspaces),
        )

        qa_rows = _load_stage_task2_qas(question_csv_path)
        if args.max_questions_per_movie and args.max_questions_per_movie > 0:
            qa_rows = qa_rows[: int(args.max_questions_per_movie)]
        total_questions += len(qa_rows)

        evaluator = HybridRAGEvaluator(
            cfg,
            setting_name=setting_name,
            article_name=movie_id,
            workspace_dir=workspace_dir,
            article_language=language,
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

        summary = _summarize_setting(
            movie_results,
            question_count=len(qa_rows),
            repeats=max(1, int(args.repeats or 1)),
        )
        movie_payload = {
            "setting": setting_name,
            "movie_id": movie_id,
            "language": language,
            "question_count": len(qa_rows),
            "repeats": max(1, int(args.repeats or 1)),
            "results": movie_results,
            "summary": summary,
        }
        json_dump_atomic(str(report_path), movie_payload)
        movie_summaries[movie_id] = {"language": language, **summary}
        all_results.extend(movie_results)

        progress_payload = {
            "setting": setting_name,
            "article_name": movie_id,
            "repeats": max(1, int(args.repeats or 1)),
            "repeat_index": max(1, int(args.repeats or 1)),
            "batch_index": movie_index,
            "batch_total": len(movie_pairs),
            "phase": "completed_movie",
            "evaluated_attempts_done": len(all_results),
            "evaluated_attempts_total": total_questions * max(1, int(args.repeats or 1)),
            "batch_question_count": len(qa_rows),
            "note": f"completed {movie_index}/{len(movie_pairs)} movies",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_dump_atomic(str(progress_path), progress_payload)
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

    overall_summary = _summarize_setting(
        all_results,
        question_count=total_questions,
        repeats=max(1, int(args.repeats or 1)),
    )
    final_payload = {
        "benchmark_root": str(benchmark_root),
        "config": str(config_path),
        "setting": setting_name,
        "retrieval_mode": retrieval_mode,
        "fusion_mode": fusion_mode,
        "movie_count": len(movie_pairs),
        "question_count": total_questions,
        "repeats": max(1, int(args.repeats or 1)),
        "eval_max_workers": max(1, int(args.eval_max_workers or 1)),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "dense_top_k": int(args.dense_top_k),
        "bm25_top_k": int(args.bm25_top_k),
        "final_top_k": int(args.final_top_k),
        "summary": overall_summary,
        "movie_summaries": movie_summaries,
        "selected_movies": manifest,
        "manifest_path": str(run_root / "manifest.json"),
    }
    json_dump_atomic(str(report_root / "summary.json"), final_payload)

    md_lines = [
        "# STAGE Task 2 Hybrid RAG Summary",
        "",
        f"- movie_count: {len(movie_pairs)}",
        f"- question_count: {total_questions}",
        f"- repeats: {max(1, int(args.repeats or 1))}",
        f"- eval_max_workers: {max(1, int(args.eval_max_workers or 1))}",
        f"- chunk_size: {int(args.chunk_size)}",
        f"- chunk_overlap: {int(args.chunk_overlap)}",
        f"- dense_top_k: {int(args.dense_top_k)}",
        f"- bm25_top_k: {int(args.bm25_top_k)}",
        f"- final_top_k: {int(args.final_top_k)}",
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
    md_lines.extend(["", "## Per Movie", ""])
    for movie_id, summary in sorted(movie_summaries.items()):
        md_lines.append(
            f"- {movie_id} ({summary.get('language', '')}): overall={summary['overall_accuracy']} pass={summary['pass_accuracy']} avg_judge={summary['avg_judge_score']}"
        )
    (report_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Finished STAGE hybrid RAG benchmark. summary=%s", report_root / "summary.json")


if __name__ == "__main__":
    main()
