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
from core.utils.config import KAGConfig
from core.utils.general_utils import (
    compress_query_for_vector_search,
    json_dump_atomic,
    parse_json_object_from_text,
    word_len,
)
from core.utils.prompt_loader import YAMLPromptLoader
from retriever.sparse_retriever import KeywordBM25Retriever

logger = logging.getLogger(__name__)

_QUESTION_TYPE_FIELDS: Tuple[str, ...] = (
    "local-or-sum",
    "attribute1",
    "attribute2",
    "ex-or-im1",
    "ex-or-im2",
)


@dataclass
class OpenEvalResult:
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
    return REPO_ROOT / "experiments" / "fiarytableqa" / "assets" / "article_workspaces_hybridrag_plain"


def _default_converted_article_root() -> Path:
    return REPO_ROOT / "experiments" / "fiarytableqa" / "assets" / "converted_articles"


def _story_title_from_slug(slug: str) -> str:
    raw = str(slug or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(piece for piece in raw.split() if piece).strip() or slug


def _convert_fiarytable_article(*, article_name: str, src_path: Path, dst_path: Path) -> None:
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected article JSON list: {src_path}")
    title = _story_title_from_slug(article_name)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        raw_id = item.get("id")
        part_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else str(idx)
        out.append(
            {
                "id": f"{article_name}_part_{part_id}",
                "title": title,
                "subtitle": f"Part {part_id}",
                "content": content,
            }
        )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump_atomic(str(dst_path), out)


def _extract_question_type_fields(metadata: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in _QUESTION_TYPE_FIELDS:
        value = str((metadata or {}).get(key, "") or "").strip()
        if value:
            out[key] = value
    return out


def _extract_question_type_tags(fields: Dict[str, str]) -> List[str]:
    tags = [f"{key}:{value}" for key, value in fields.items() if str(value or "").strip()]
    attr1 = str(fields.get("attribute1", "") or "").strip()
    attr2 = str(fields.get("attribute2", "") or "").strip()
    if attr1:
        combo = attr1 if not attr2 else f"{attr1} + {attr2}"
        tags.append(f"attribute_combo:{combo}")
    ex1 = str(fields.get("ex-or-im1", "") or "").strip()
    ex2 = str(fields.get("ex-or-im2", "") or "").strip()
    if ex1:
        combo = ex1 if not ex2 else f"{ex1} + {ex2}"
        tags.append(f"explicitness_combo:{combo}")
    return tags


def _format_reference_answers(answers: Sequence[str]) -> str:
    cleaned: List[str] = []
    seen = set()
    for ans in answers or []:
        text = str(ans or "").strip()
        norm = text.lower()
        if not text or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(text)
    if not cleaned:
        return "(no reference answer)"
    return "Acceptable answer variants (any one is sufficient): " + " || ".join(cleaned)


def _load_fiarytable_qas(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str((row or {}).get("question", "") or "").strip()
            if not question:
                continue
            answers: List[str] = []
            for key in ("answer1", "answer2", "answer3", "answer4", "answer5", "answer6"):
                value = str((row or {}).get(key, "") or "").strip()
                if value:
                    answers.append(value)
            metadata = dict(row or {})
            question_type_fields = _extract_question_type_fields(metadata)
            rows.append(
                {
                    "question_id": str((row or {}).get("question_id", "") or "").strip() or f"q{len(rows)}",
                    "question": question,
                    "reference_answers": answers,
                    "reference_answer": _format_reference_answers(answers),
                    "metadata": metadata,
                    "question_type_fields": question_type_fields,
                    "question_type_tags": _extract_question_type_tags(question_type_fields),
                }
            )
    return rows


def _load_manifest_article_pairs(manifest_path: Path) -> List[Tuple[str, Path, Path]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    out: List[Tuple[str, Path, Path]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        article_name = str(item.get("article_name", "") or "").strip()
        article_path = _resolve_cli_path(str(item.get("article_json", "") or ""))
        question_path = _resolve_cli_path(str(item.get("question_csv", "") or ""))
        if not article_name or not article_path.exists() or not question_path.exists():
            continue
        out.append((article_name, article_path, question_path))
    return out


def _build_base_cfg(config_path: Path) -> KAGConfig:
    cfg = KAGConfig.from_yaml(str(config_path))
    cfg.global_.language = "en"
    cfg.global_.locale = "en"
    cfg.global_config.language = "en"
    cfg.global_config.locale = "en"
    return cfg


def _build_article_cfg(
    base_cfg: KAGConfig,
    *,
    workspace_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    dense_top_k: int,
    bm25_top_k: int,
    final_top_k: int,
) -> KAGConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.global_.language = "en"
    cfg.global_.locale = "en"
    cfg.global_config.language = "en"
    cfg.global_config.locale = "en"
    cfg.document_processing.chunk_size = int(chunk_size)
    cfg.document_processing.chunk_overlap = int(chunk_overlap)
    cfg.document_processing.dense_top_k = int(dense_top_k)
    cfg.document_processing.bm25_top_k = int(bm25_top_k)
    cfg.document_processing.final_top_k = int(final_top_k)
    cfg.storage.vector_store_path = str(workspace_dir / "vector_store")
    return cfg


def _build_signature(cfg: KAGConfig, *, retrieval_mode: str, fusion_mode: str) -> Dict[str, Any]:
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


def _chunk_article_documents(
    *,
    article_name: str,
    converted_article_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    parts = json.loads(converted_article_path.read_text(encoding="utf-8"))
    if not isinstance(parts, list):
        raise ValueError(f"Expected converted article list: {converted_article_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(10, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
        length_function=lambda text: word_len(text, lang="auto"),
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
        keep_separator=True,
    )

    chunks: List[Dict[str, Any]] = []
    for doc_index, item in enumerate(parts, start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        document_id = str(item.get("id", "") or f"{article_name}_part_{doc_index}")
        title = str(item.get("title", "") or "").strip()
        subtitle = str(item.get("subtitle", "") or "").strip()
        header = "\n".join(part for part in [title, subtitle] if part).strip()
        payload_text = f"{header}\n{content}".strip() if header else content
        doc_parts = splitter.split_text(payload_text) if word_len(payload_text, lang="auto") > int(chunk_size) else [payload_text]
        for chunk_index, chunk_text in enumerate(doc_parts, start=1):
            text = str(chunk_text or "").strip()
            if not text:
                continue
            chunk_id = f"{document_id}_chunk_{chunk_index:03d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "content": text,
                    "metadata": {
                        "article_name": article_name,
                        "title": title,
                        "subtitle": subtitle,
                        "document_id": document_id,
                        "document_order": doc_index,
                        "chunk_id": chunk_id,
                        "chunk_order": chunk_index,
                    },
                }
            )
    return chunks


def _build_workspace_assets(
    *,
    cfg: KAGConfig,
    workspace_dir: Path,
    converted_article_path: Path,
    article_name: str,
    retrieval_mode: str,
    fusion_mode: str,
    rebuild: bool,
) -> Dict[str, Any]:
    signature = _build_signature(cfg, retrieval_mode=retrieval_mode, fusion_mode=fusion_mode)
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

    chunks = _chunk_article_documents(
        article_name=article_name,
        converted_article_path=converted_article_path,
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
        "article_name": article_name,
        "converted_article_path": str(converted_article_path),
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
        document_id = str(md.get("document_id", "") or "").strip()
        chunk_id = str(item.get("id", "") or "").strip()
        header_parts = [part for part in [title, subtitle] if part]
        header = " / ".join(header_parts) if header_parts else document_id or chunk_id
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


def _build_answer_prompt(*, question: str, evidence_text: str) -> str:
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


class HybridRAGArticleRuntime:
    def __init__(self, cfg: KAGConfig, *, workspace_dir: Path) -> None:
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.chunks = _load_workspace_chunks(workspace_dir)
        self.chunk_map = {str(item.get("id", "") or ""): item for item in self.chunks}
        self.vector_store = VectorStore(cfg, category="chunks")
        self.bm25 = KeywordBM25Retriever(
            [
                LCDocument(
                    page_content=str(item.get("content", "") or ""),
                    metadata=dict(item.get("metadata") or {}) | {"chunk_id": str(item.get("id", "") or "")},
                )
                for item in self.chunks
            ],
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
        _close_vector_store(self.vector_store)


class HybridRAGThreadLocal:
    def __init__(self, cfg: KAGConfig, *, workspace_dir: Path) -> None:
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _build_state(self) -> Dict[str, Any]:
        runtime = HybridRAGArticleRuntime(self.cfg, workspace_dir=self.workspace_dir)
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
    ) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.article_name = article_name
        self.tlocal = HybridRAGThreadLocal(cfg, workspace_dir=workspace_dir)

    def evaluate_row(self, row: Dict[str, Any], run_index: int) -> OpenEvalResult:
        started = time.time()
        state = self.tlocal.state()
        retrieval = state["runtime"].retrieve(str(row.get("question", "") or ""))
        prompt = _build_answer_prompt(
            question=str(row.get("question", "") or ""),
            evidence_text=str(retrieval.get("evidence_text", "") or ""),
        )
        answer_raw = ""
        error_text = ""
        judge_result = {
            "is_correct": False,
            "score": 0.0,
            "reason": "evaluation_not_run",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
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

        return OpenEvalResult(
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
                "raw_answer": answer_raw,
                "judge": judge_result,
                "reference_answers": list(row.get("reference_answers") or []),
                "question_metadata": dict(row.get("metadata") or {}),
                "question_type_fields": dict(row.get("question_type_fields") or {}),
                "question_type_tags": list(row.get("question_type_tags") or []),
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
    type_breakdown: Dict[str, Dict[str, Any]] = {}
    for field in _QUESTION_TYPE_FIELDS:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            fields = item.get("question_type_fields") or {}
            if not isinstance(fields, dict):
                continue
            label = str(fields.get(field, "") or "").strip()
            if not label:
                continue
            buckets.setdefault(label, []).append(item)
        if not buckets:
            continue
        field_summary: Dict[str, Any] = {}
        for label, bucket in sorted(buckets.items()):
            bucket_question_count = len({(str(x.get("article_name", "") or ""), str(x.get("question_id", "") or "")) for x in bucket})
            field_summary[label] = _summarize_setting_base(bucket, question_count=bucket_question_count, repeats=repeats)
        type_breakdown[field] = field_summary
    if type_breakdown:
        summary["type_breakdown"] = type_breakdown
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openai_quality_stable.yaml")
    parser.add_argument(
        "--manifest-path",
        default="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/manifest.json",
    )
    parser.add_argument("--limit-articles", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-max-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=8)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--rebuild-workspaces", action="store_true")
    parser.add_argument("--workspace-asset-root", default="experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-questions-per-article", type=int, default=0)
    parser.add_argument("--skip-existing-reports", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = _resolve_cli_path(args.config)
    manifest_path = _resolve_cli_path(args.manifest_path)
    workspace_asset_root = _resolve_cli_path(args.workspace_asset_root) if str(args.workspace_asset_root or "").strip() else _default_workspace_asset_root()
    workspace_asset_root.mkdir(parents=True, exist_ok=True)
    converted_article_root = _default_converted_article_root()
    converted_article_root.mkdir(parents=True, exist_ok=True)

    run_name = str(args.run_name or "").strip() or time.strftime("fiarytableqa_manifest100_hybridrag_plain_%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "experiments" / "fiarytableqa" / "runs" / run_name
    report_root = run_root / "reports"
    for path in (run_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    article_pairs = _load_manifest_article_pairs(manifest_path)
    if args.limit_articles and args.limit_articles > 0:
        article_pairs = article_pairs[: int(args.limit_articles)]

    manifest = [
        {
            "article_name": article_name,
            "article_json": str(article_path),
            "question_csv": str(question_path),
        }
        for article_name, article_path, question_path in article_pairs
    ]
    json_dump_atomic(str(run_root / "manifest.json"), manifest)

    base_cfg = _build_base_cfg(config_path)
    setting_name = "hybrid_rag_plain"
    retrieval_mode = "hybrid_rag_plain"
    fusion_mode = "rrf"

    experiment_lines = [
        "# FiarytableQA Hybrid RAG Experiment",
        "",
        f"- run_name: {run_name}",
        f"- setting: {setting_name}",
        f"- retrieval_mode: {retrieval_mode}",
        f"- fusion_mode: {fusion_mode}",
        f"- config: {config_path}",
        f"- manifest_path: {manifest_path}",
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
        "## Selected Articles",
        "",
    ]
    for article_name, article_path, question_path in article_pairs:
        experiment_lines.append(f"- {article_name}: article={article_path} qa={question_path}")
    (run_root / "experiment.md").write_text("\n".join(experiment_lines) + "\n", encoding="utf-8")

    all_results: List[Dict[str, Any]] = []
    article_summaries: Dict[str, Any] = {}
    total_questions = 0
    progress_path = report_root / "progress.json"

    for article_index, (article_name, article_src_path, question_csv_path) in enumerate(article_pairs, start=1):
        report_path = report_root / f"{article_name}.json"
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
                    article_summaries[article_name] = existing_summary
                    all_results.extend([item for item in existing_results if isinstance(item, dict)])
                    continue

        converted_article_path = converted_article_root / f"{article_name}.json"
        if not converted_article_path.exists():
            _convert_fiarytable_article(
                article_name=article_name,
                src_path=article_src_path,
                dst_path=converted_article_path,
            )

        workspace_dir = workspace_asset_root / article_name
        cfg = _build_article_cfg(
            base_cfg,
            workspace_dir=workspace_dir,
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
            dense_top_k=int(args.dense_top_k),
            bm25_top_k=int(args.bm25_top_k),
            final_top_k=int(args.final_top_k),
        )
        _build_workspace_assets(
            cfg=cfg,
            workspace_dir=workspace_dir,
            converted_article_path=converted_article_path,
            article_name=article_name,
            retrieval_mode=retrieval_mode,
            fusion_mode=fusion_mode,
            rebuild=bool(args.rebuild_workspaces),
        )

        qa_rows = _load_fiarytable_qas(question_csv_path)
        if args.max_questions_per_article and args.max_questions_per_article > 0:
            qa_rows = qa_rows[: int(args.max_questions_per_article)]
        total_questions += len(qa_rows)

        evaluator = HybridRAGEvaluator(
            cfg,
            setting_name=setting_name,
            article_name=article_name,
            workspace_dir=workspace_dir,
        )
        try:
            article_results = _evaluate_setting(
                rows=qa_rows,
                evaluator=evaluator,
                repeats=max(1, int(args.repeats or 1)),
                max_workers=max(1, int(args.eval_max_workers or 1)),
            )
        finally:
            evaluator.close()

        summary = _summarize_setting(
            article_results,
            question_count=len(qa_rows),
            repeats=max(1, int(args.repeats or 1)),
        )
        article_payload = {
            "setting": setting_name,
            "article_name": article_name,
            "question_count": len(qa_rows),
            "repeats": max(1, int(args.repeats or 1)),
            "results": article_results,
            "summary": summary,
        }
        json_dump_atomic(str(report_path), article_payload)
        article_summaries[article_name] = summary
        all_results.extend(article_results)

        progress_payload = {
            "setting": setting_name,
            "article_name": article_name,
            "repeats": max(1, int(args.repeats or 1)),
            "repeat_index": max(1, int(args.repeats or 1)),
            "batch_index": article_index,
            "batch_total": len(article_pairs),
            "phase": "completed_article",
            "evaluated_attempts_done": len(all_results),
            "evaluated_attempts_total": total_questions * max(1, int(args.repeats or 1)),
            "batch_question_count": len(qa_rows),
            "note": f"completed {article_index}/{len(article_pairs)} articles",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_dump_atomic(str(progress_path), progress_payload)
        logger.info(
            "[%d/%d] article=%s overall=%.4f pass=%.4f avg_judge=%.4f",
            article_index,
            len(article_pairs),
            article_name,
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
        "config": str(config_path),
        "manifest_path": str(manifest_path),
        "setting": setting_name,
        "retrieval_mode": retrieval_mode,
        "fusion_mode": fusion_mode,
        "article_count": len(article_pairs),
        "question_count": total_questions,
        "repeats": max(1, int(args.repeats or 1)),
        "eval_max_workers": max(1, int(args.eval_max_workers or 1)),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "dense_top_k": int(args.dense_top_k),
        "bm25_top_k": int(args.bm25_top_k),
        "final_top_k": int(args.final_top_k),
        "summary": overall_summary,
        "article_summaries": article_summaries,
        "selected_articles": manifest,
    }
    json_dump_atomic(str(report_root / "summary.json"), final_payload)

    md_lines = [
        "# FiarytableQA Hybrid RAG Summary",
        "",
        f"- article_count: {len(article_pairs)}",
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
        "## Selected Articles",
        "",
    ]
    for article_name, article_path, question_path in article_pairs:
        md_lines.append(f"- {article_name}: article={article_path} qa={question_path}")
    md_lines.extend(["", "## Per Article", ""])
    for article_name, summary in sorted(article_summaries.items()):
        md_lines.append(
            f"- {article_name}: overall={summary['overall_accuracy']} pass={summary['pass_accuracy']} avg_judge={summary['avg_judge_score']}"
        )

    type_breakdown = overall_summary.get("type_breakdown") if isinstance(overall_summary.get("type_breakdown"), dict) else {}
    if type_breakdown:
        md_lines.extend(["", "## Type Breakdown", ""])
        for field, field_payload in sorted(type_breakdown.items()):
            md_lines.append(f"### {field}")
            md_lines.append("")
            for label, bucket_summary in sorted((field_payload or {}).items()):
                md_lines.append(
                    f"- {label}: overall={bucket_summary['overall_accuracy']} pass={bucket_summary['pass_accuracy']} avg_judge={bucket_summary['avg_judge_score']}"
                )
            md_lines.append("")

    (report_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Finished FiarytableQA hybrid RAG benchmark. summary=%s", report_root / "summary.json")


if __name__ == "__main__":
    main()
