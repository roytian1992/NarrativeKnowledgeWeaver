from __future__ import annotations

import argparse
import ast
import copy
import csv
import fcntl
import json
import logging
import os
import pickle
import re
import signal
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document as LCDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent, _split_bm25_documents
from core.builder.graph_builder import KnowledgeGraphBuilder
from core.builder.narrative_graph_builder import NarrativeGraphBuilder
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.models.data import Document as KAGDocument
from core.model_providers.openai_embedding import OpenAIEmbeddingModel
from core.model_providers.openai_llm import OpenAILLM
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.strategy_training.online_strategy_training_runner import OnlineStrategyTrainingRunner
from core.strategy_training.strategy_cluster_manager import StrategyTemplateClusterManager
from core.strategy_training.strategy_runtime_assets import StrategyRuntimeAssetManager
from core.strategy_training.strategy_training_runner import StrategyMemoryTrainingRunner
from core.utils.config import KAGConfig, _apply_global_locale_paths
from core.utils.general_utils import dump_json, json_dump_atomic, load_json, word_len
from core.utils.graph_query_utils import GraphQueryUtils
from core.utils.prompt_loader import YAMLPromptLoader
from retriever.sparse_retriever import KeywordBM25Retriever


CHOICE_RE = re.compile(r"\(([A-Z])\)\s*")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    setting: str
    article_name: str
    run_index: int
    question_id: str
    question: str
    reference_choice: str
    reference_answer: str
    predicted_choice: str
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
            "reference_choice": self.reference_choice,
            "reference_answer": self.reference_answer,
            "predicted_choice": self.predicted_choice,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            **self.extra,
        }


def _extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    match = JSON_RE.search(raw)
    if match:
        candidates.append(match.group(0))
    for item in candidates:
        try:
            payload = json.loads(item)
        except Exception:
            try:
                payload = ast.literal_eval(item)
            except Exception:
                continue
        if isinstance(payload, dict):
            return payload
    return None


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = load_json(str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return payload


def _load_article_qas(path: Path, *, with_answer: bool = True) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = str((row or {}).get("question", "") or "").strip()
            answer_choice = str((row or {}).get("answer_choice", "") or "").strip().upper()
            answer_text = str((row or {}).get("answer_text", "") or "").strip()
            if not question:
                continue
            payload = {
                "question_id": f"q{idx}",
                "question": question,
                "answer_choice": answer_choice,
                "answer_text": answer_text,
            }
            if not with_answer:
                payload.pop("answer_choice", None)
                payload.pop("answer_text", None)
            rows.append(payload)
    return rows


def _parse_choices(question_text: str) -> Dict[str, Any]:
    raw = str(question_text or "").strip()
    matches = list(CHOICE_RE.finditer(raw))
    if not matches:
        return {"question_stem": raw, "choices": {}, "choice_order": []}
    stem = raw[: matches[0].start()].strip()
    choices: Dict[str, str] = {}
    order: List[str] = []
    for idx, match in enumerate(matches):
        label = str(match.group(1) or "").strip().upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        text = raw[start:end].strip()
        if not label or not text:
            continue
        choices[label] = text
        order.append(label)
    return {"question_stem": stem or raw, "choices": choices, "choice_order": order}


def _format_mcq_for_agent(question_text: str) -> str:
    parsed = _parse_choices(question_text)
    stem = str(parsed.get("question_stem", "") or "").strip()
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
    order = list(parsed.get("choice_order") or [])
    lines = [
        "Answer the following multiple-choice question using retrieval.",
        "",
        "Return JSON only:",
        '{"answer_choice":"A","answer_text":"...","evidence":"..."}',
        "",
        "Rules:",
        "- Choose exactly one option label.",
        "- `answer_choice` must be one of the provided labels.",
        "- `answer_text` should restate the chosen option, not the full reasoning.",
        "- `evidence` should be brief.",
        "- Compare the retrieved evidence against every option before choosing.",
        "- Prefer the option that is most directly supported and requires the fewest extra assumptions.",
        "- Do not invent motives, warnings, implications, or relationships unless the evidence supports them.",
        "- If the options are close, retrieve more evidence targeted at the phrases that distinguish the options.",
        "",
        "Question:",
        stem or str(question_text or "").strip(),
    ]
    if choices:
        lines.extend(["", "Choices:"])
        for label in order:
            value = str(choices.get(label, "") or "").strip()
            if value:
                lines.append(f"{label}. {value}")
    return "\n".join(lines).strip()


def _format_open_question_for_agent(question_text: str) -> str:
    parsed = _parse_choices(question_text)
    stem = str(parsed.get("question_stem", "") or "").strip() or str(question_text or "").strip()
    lines = [
        "Answer the following question using retrieval.",
        "",
        "Return JSON only:",
        '{"answer_text":"...","evidence":"...","confidence":0.72}',
        "",
        "Rules:",
        "- You do not have access to answer options.",
        "- `answer_text` should directly answer the question in one short sentence or phrase.",
        "- `evidence` should be brief and grounded in retrieved evidence.",
        "- `confidence` must be a number between 0 and 1.",
        "- Do not mention option labels such as A/B/C/D.",
        "- Do not pad the answer with speculative alternatives.",
        "",
        "Question:",
        stem,
    ]
    return "\n".join(lines).strip()


def _normalize_choice_from_answer_text(answer_text: str, choices: Dict[str, str]) -> str:
    raw = str(answer_text or "").strip()
    if not raw:
        return ""
    lower = raw.lower()
    matched: List[str] = []
    for label, text in choices.items():
        norm = str(text or "").strip().lower()
        if norm and norm in lower:
            matched.append(label)
    if len(matched) == 1:
        return matched[0]
    return ""


def _extract_nested_answer_texts(answer_text: str, *, max_depth: int = 3) -> List[str]:
    seed = str(answer_text or "").strip()
    if not seed:
        return []

    out: List[str] = []
    seen: set[str] = set()

    def push(text: Any) -> None:
        raw = str(text or "").strip()
        if raw and raw not in seen:
            seen.add(raw)
            out.append(raw)

    def walk(obj: Any, depth: int) -> None:
        if depth > max_depth or obj is None:
            return
        if isinstance(obj, str):
            push(obj)
            parsed = _extract_json_object(obj)
            if isinstance(parsed, dict):
                walk(parsed, depth + 1)
            else:
                try:
                    literal = ast.literal_eval(obj)
                except Exception:
                    literal = None
                if literal is not None and literal is not obj:
                    walk(literal, depth + 1)
            return
        if isinstance(obj, dict):
            for key in (
                "content",
                "answer",
                "final_answer",
                "predicted_answer",
                "answer_text",
                "text",
                "output",
            ):
                if key in obj:
                    walk(obj.get(key), depth + 1)
            return
        if isinstance(obj, list):
            for item in obj[:8]:
                walk(item, depth + 1)

    walk(seed, 0)
    return out


def _extract_choice_label_heuristic(text: str, valid_labels: List[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    valid = {str(x or "").strip().upper() for x in valid_labels if str(x or "").strip()}
    payload = _extract_json_object(raw)
    if payload:
        for key in ("answer_choice", "choice", "option", "label"):
            value = str(payload.get(key, "") or "").strip().upper()
            if value in valid:
                return value
    patterns = [
        r"answer_choice\s*[:=]\s*['\"]?([A-Z])['\"]?",
        r"choice\s*[:=]\s*['\"]?([A-Z])['\"]?",
        r"option\s*[:=]\s*['\"]?([A-Z])['\"]?",
        r"^\s*([A-Z])[\.\):\-\s]",
        r"\boption\s+([A-Z])\b",
        r"\bchoice\s+([A-Z])\b",
        r"\banswer\s+is\s+([A-Z])\b",
        r"\b(?:final|best|correct)\s+answer\s*[:=]?\s*([A-Z])\b",
        r"\b(?:choose|pick|select)\s+([A-Z])\b",
        r"[\[\(\{'\"]([A-Z])[\]\)\}'\"]",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        value = str(m.group(1) or "").strip().upper()
        if value in valid:
            return value
    if len(raw) == 1 and raw.upper() in valid:
        return raw.upper()
    return ""


def _extract_choice_from_choice_tool_output(answer_text: str, valid_labels: List[str]) -> str:
    raw = str(answer_text or "").strip()
    if not raw:
        return ""
    direct = _extract_choice_label_heuristic(raw, valid_labels)
    if direct:
        return direct
    patterns = [
        r"\[Recommended Choice\]\s*([A-Z])\.",
        r"\[Suggested Choice\]\s*([A-Z])\b",
        r"selected=([A-Z])\b",
        r"recommended[_ ]choice[^A-Z]*([A-Z])\b",
    ]
    valid = {str(x or "").strip().upper() for x in valid_labels if str(x or "").strip()}
    for pattern in patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        value = str(m.group(1) or "").strip().upper()
        if value in valid:
            return value
    return ""


def _extract_llm_text(result: Any) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        content = getattr(item, "content", None)
        if isinstance(content, str):
            return content.strip()
    if isinstance(result, str):
        return result.strip()
    return str(result or "").strip()


def _has_error_payload(answer_text: str) -> bool:
    payload = _extract_json_object(answer_text)
    return isinstance(payload, dict) and isinstance(payload.get("error"), dict)


def _looks_like_tool_call_payload(answer_text: str) -> bool:
    payload = _extract_json_object(answer_text)
    if not isinstance(payload, dict):
        return False
    if any(key in payload for key in ("answer_choice", "choice", "option", "label")):
        return False
    tool_name = str(payload.get("tool_name", "") or "").strip()
    if not tool_name:
        return False
    return "tool_arguments" in payload or "arguments" in payload


def _extract_answer_confidence(answer_text: str) -> Optional[float]:
    payload = _extract_json_object(answer_text)
    if isinstance(payload, dict):
        for key in ("confidence", "answer_confidence", "score"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                conf = float(value)
            except Exception:
                continue
            return max(0.0, min(1.0, conf))
    raw = str(answer_text or "").strip()
    if raw:
        for pattern in (
            r"confidence\s*[:=]\s*['\"]?([01](?:\.\d+)?)['\"]?",
            r"answer_confidence\s*[:=]\s*['\"]?([01](?:\.\d+)?)['\"]?",
            r"score\s*[:=]\s*['\"]?([01](?:\.\d+)?)['\"]?",
        ):
            m = re.search(pattern, raw, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                conf = float(m.group(1))
            except Exception:
                continue
            return max(0.0, min(1.0, conf))
    return None


def _ensure_answer_confidence(answer_text: str, *, default: float = 0.62) -> str:
    text = str(answer_text or "").strip()
    if not text:
        return text
    if _extract_answer_confidence(text) is not None:
        return text
    try:
        parsed = json.loads(correct_json_format(text))
    except Exception:
        return text
    if not isinstance(parsed, dict):
        return text
    parsed["confidence"] = max(0.0, min(1.0, float(default)))
    return json.dumps(parsed, ensure_ascii=False)


def _is_low_confidence(confidence: Optional[float], *, threshold: float = 0.5) -> bool:
    if confidence is None:
        return True
    try:
        return float(confidence) < float(threshold)
    except Exception:
        return True


def _format_mcq_for_agent_with_retrieval_guard(question_text: str) -> str:
    base = _format_mcq_for_agent(question_text)
    extra = "\n".join(
        [
            "",
            "Mandatory retrieval guard:",
            "- You must call at least one retrieval tool before answering.",
            "- Do not answer from prior knowledge or intuition.",
            "- If you have not retrieved evidence yet, continue retrieving instead of answering.",
            "- For inference, motive, warning, implication, attitude, or allusion questions, retrieve evidence that distinguishes the competing options before answering.",
            "- Entity summaries alone are usually insufficient when the answer depends on a scene, interaction, or implied meaning.",
        ]
    )
    return f"{base}{extra}"


def _format_open_question_for_agent_with_retrieval_guard(question_text: str) -> str:
    base = _format_open_question_for_agent(question_text)
    extra = "\n".join(
        [
            "",
            "Mandatory retrieval guard:",
            "- You must call at least one retrieval tool before answering.",
            "- Do not answer from prior knowledge or intuition.",
            "- If you have not retrieved evidence yet, continue retrieving instead of answering.",
            "- For inference, motive, warning, implication, attitude, or allusion questions, retrieve evidence that clarifies the scene or storyline before answering.",
            "- Entity summaries alone are usually insufficient when the answer depends on a scene, interaction, or implied meaning.",
        ]
    )
    return f"{base}{extra}"


def _extract_semantic_answer_text(answer_text: str) -> str:
    payload = _extract_json_object(answer_text)
    if isinstance(payload, dict):
        for key in ("answer_text", "answer", "final_answer", "response", "prediction"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
    return str(answer_text or "").strip()


def _trim_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _summarize_tool_uses_for_finalization(tool_uses: List[Dict[str, Any]], *, max_items: int = 6) -> str:
    lines: List[str] = []
    for idx, item in enumerate(tool_uses[: max(1, int(max_items or 6))], start=1):
        tool_name = str((item or {}).get("tool_name", "") or "").strip() or "unknown_tool"
        arguments = _trim_text((item or {}).get("tool_arguments", ""), limit=300)
        output = _trim_text((item or {}).get("tool_output", ""), limit=1200)
        lines.append(f"[Tool {idx}] {tool_name}")
        if arguments:
            lines.append(f"Arguments: {arguments}")
        if output:
            lines.append(f"Output: {output}")
    return "\n".join(lines).strip()


def _is_attribute_evaluation_question(question_text: str) -> bool:
    stem = str(_parse_choices(question_text).get("question_stem", "") or "").strip().lower()
    if not stem:
        stem = str(question_text or "").strip().lower()
    cues = (
        "how good is",
        "which word doesn't describe",
        "which word does not describe",
        "best describes",
        "best describe",
        "best describes the",
        "what best describes",
        "character",
        "attitude toward",
        "opinion about",
        "most likely:",
    )
    return any(cue in stem for cue in cues)


class ChoiceExtractor:
    def __init__(self, cfg: KAGConfig) -> None:
        self.cfg = cfg

    def choose(self, *, question_text: str, answer_text: str) -> str:
        parsed = _parse_choices(question_text)
        order = list(parsed.get("choice_order") or [])
        if not order:
            return ""
        for candidate_text in _extract_nested_answer_texts(answer_text):
            payload = _extract_json_object(candidate_text)
            if isinstance(payload, dict) and "error" in payload:
                continue
            heuristic = _extract_choice_label_heuristic(candidate_text, order)
            if heuristic:
                return heuristic
            from_answer = _normalize_choice_from_answer_text(candidate_text, parsed.get("choices") or {})
            if from_answer:
                return from_answer
        return ""


def _rrf_merge(
    doc_hits: List[Any],
    sent_hits: List[Any],
    bm25_hits: List[LCDocument],
    *,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    channels: List[Tuple[str, float, List[Any]]] = [
        ("vector_document", 1.0, doc_hits or []),
        ("vector_sentence", 0.9, sent_hits or []),
        ("bm25", 1.1, bm25_hits or []),
    ]
    k0 = 60.0
    for channel_name, weight, hits in channels:
        for rank, hit in enumerate(hits, start=1):
            if hasattr(hit, "content"):
                content = str(getattr(hit, "content", "") or "").strip()
                metadata = dict(getattr(hit, "metadata", None) or {})
                item_id = str(getattr(hit, "id", "") or metadata.get("chunk_id") or "")
            else:
                content = str(getattr(hit, "page_content", "") or "").strip()
                metadata = dict(getattr(hit, "metadata", None) or {})
                item_id = str(metadata.get("chunk_id") or metadata.get("document_id") or "")
            if not content:
                continue
            if not item_id:
                item_id = f"{channel_name}:{rank}:{hash(content)}"
            score = weight / (k0 + float(rank))
            row = fused.get(item_id)
            if row is None:
                fused[item_id] = {
                    "id": item_id,
                    "channel": channel_name,
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                    "channels": [channel_name],
                }
            else:
                row["score"] = float(row.get("score", 0.0)) + score
                if channel_name not in row["channels"]:
                    row["channels"].append(channel_name)
                if len(content) > len(str(row.get("content", "") or "")):
                    row["content"] = content
                    row["metadata"] = metadata
    rows = sorted(
        fused.values(),
        key=lambda item: (float(item.get("score", 0.0)), len(str(item.get("content", "") or ""))),
        reverse=True,
    )
    return rows[: max(1, int(top_k or 8))]


def _format_evidence_item(idx: int, item: Dict[str, Any]) -> str:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return "\n".join(
        [
            f"[{idx}] channel={item.get('channel', '')}",
            f"document_id={meta.get('document_id', '')}",
            f"title={meta.get('title') or meta.get('source_title') or meta.get('doc_title') or ''}",
            "content:",
            str(item.get("content", "") or "").strip(),
        ]
    )


def _set_global_language(cfg: KAGConfig, language: str) -> None:
    cfg.global_.language = language
    cfg.global_.locale = language
    cfg.global_config.language = language
    cfg.global_config.locale = language
    _apply_global_locale_paths(cfg.global_)
    _apply_global_locale_paths(cfg.global_config)


def _build_article_config(
    base_cfg: KAGConfig,
    *,
    workspace_dir: Path,
    runtime_library_path: Optional[Path] = None,
    runtime_tool_metadata_dir: Optional[Path] = None,
    strategy_enabled: bool = False,
    subagent_enabled: bool = False,
    online_runtime_mode: bool = False,
    strategy_read_enabled: Optional[bool] = None,
    runtime_routing_note_enabled: Optional[bool] = None,
) -> KAGConfig:
    cfg = copy.deepcopy(base_cfg)
    _set_global_language(cfg, "en")
    cfg.global_.doc_type = "general"
    cfg.global_config.doc_type = "general"
    cfg.global_.aggregation_mode = "narrative"
    cfg.global_config.aggregation_mode = "narrative"

    cfg.document_processing.max_workers = min(16, int(cfg.document_processing.max_workers or 16))
    cfg.document_processing.reset_vector_collections = True
    cfg.knowledge_graph_builder.max_workers = min(16, int(cfg.knowledge_graph_builder.max_workers or 16))
    cfg.narrative_graph_builder.max_workers = min(16, int(cfg.narrative_graph_builder.max_workers or 16))

    cfg.knowledge_graph_builder.file_path = str(workspace_dir / "knowledge_graph")
    cfg.narrative_graph_builder.file_path = str(workspace_dir / "narrative_graph")
    community_dir = workspace_dir / "community_graph"
    cfg.community_graph_builder.file_path = str(community_dir) if community_dir.exists() else ""
    cfg.storage.graph_store_path = str(workspace_dir / "knowledge_graph" / "graph_runtime_langgraph.pkl")
    cfg.storage.vector_store_path = str(workspace_dir / "vector_store")
    cfg.storage.sql_database_path = str(workspace_dir / "sql")

    cfg.extraction_memory.enabled = False
    cfg.strategy_memory.enabled = strategy_enabled
    runtime_routing_enabled = (
        bool(runtime_routing_note_enabled)
        if runtime_routing_note_enabled is not None
        else bool(getattr(cfg.strategy_memory, "runtime_routing_note_enabled", False))
    )
    cfg.strategy_memory.runtime_routing_note_enabled = runtime_routing_enabled
    cfg.strategy_memory.read_enabled = (
        bool(strategy_read_enabled)
        if strategy_read_enabled is not None
        else bool(strategy_enabled or runtime_routing_enabled)
    )
    cfg.strategy_memory.subagent_enabled = subagent_enabled
    cfg.strategy_memory.online_runtime_mode = bool(online_runtime_mode)
    cfg.strategy_memory.subagent_max_branches = 5
    cfg.strategy_memory.min_sampling_branches = (
        max(2, int(getattr(cfg.strategy_memory, "min_sampling_branches", 5) or 5))
        if cfg.strategy_memory.read_enabled
        else 1
    )
    if runtime_library_path is not None:
        cfg.strategy_memory.library_path = str(runtime_library_path)
    if runtime_tool_metadata_dir is not None:
        cfg.strategy_memory.tool_metadata_runtime_dir = str(runtime_tool_metadata_dir)
    if runtime_library_path is not None:
        cfg.strategy_memory.source_question_hint_path = str(runtime_library_path.parent / "source_question_hints.json")
    return cfg


def _article_interaction_artifact_paths(cfg: KAGConfig) -> Dict[str, Path]:
    workspace_dir = Path(str(cfg.knowledge_graph_builder.file_path)).parent
    interaction_dir = workspace_dir / "interactions"
    sql_dir = workspace_dir / "sql"
    return {
        "workspace_dir": workspace_dir,
        "interaction_dir": interaction_dir,
        "interaction_json_path": interaction_dir / "interaction_results.json",
        "interaction_list_json_path": interaction_dir / "interaction_records_list.json",
        "sql_db_path": sql_dir / "Interaction.db",
    }


def _workspace_missing_artifacts(workspace_dir: Path) -> List[str]:
    required_paths = [
        workspace_dir / "build_marker.json",
        workspace_dir / "interactions" / "interaction_results.json",
        workspace_dir / "interactions" / "interaction_records_list.json",
        workspace_dir / "sql" / "Interaction.db",
    ]
    missing: List[str] = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))
    return missing


def _load_doc2chunks_for_vector_rebuild(kg_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    doc2chunks_path = kg_dir / "doc2chunks.json"
    if doc2chunks_path.exists():
        payload = load_json(str(doc2chunks_path))
        if isinstance(payload, dict) and payload:
            return payload, doc2chunks_path

    all_chunks_path = kg_dir / "all_document_chunks.json"
    if not all_chunks_path.exists():
        raise FileNotFoundError(f"Missing vector rebuild source under {kg_dir}")

    payload = load_json(str(all_chunks_path))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid all_document_chunks payload: {all_chunks_path}")

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        document_id = str(metadata.get("document_id", "") or "").strip()
        if not document_id:
            continue
        pack = grouped.setdefault(
            document_id,
            {
                "document_metadata": {},
                "chunks": [],
            },
        )
        if not pack["document_metadata"]:
            doc_md = dict(metadata)
            doc_md["document_id"] = document_id
            pack["document_metadata"] = doc_md
        pack["chunks"].append(
            {
                "id": item.get("id"),
                "content": item.get("content"),
                "metadata": metadata,
            }
        )

    if not grouped:
        raise ValueError(f"No valid document groups found in {all_chunks_path}")
    return grouped, all_chunks_path


def _persist_vector_stores_from_doc2chunks(cfg: KAGConfig, doc2chunks: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    document_vector_store = VectorStore(cfg, category="document")
    sentence_vector_store = VectorStore(cfg, category="sentence")
    document_vector_store.delete_collection()
    document_vector_store._initialize()
    sentence_vector_store.delete_collection()
    sentence_vector_store._initialize()

    sentence_chunk_size = int(getattr(cfg.document_processing, "sentence_chunk_size", 200) or 200)
    sentence_chunk_overlap = int(getattr(cfg.document_processing, "sentence_chunk_overlap", 50) or 50)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(10, sentence_chunk_size),
        chunk_overlap=max(0, sentence_chunk_overlap),
        length_function=lambda t: word_len(t, lang="auto"),
        separators=[
            "\n\n", "### ", "## ", "# ",
            "\n",
            "。", "！", "？", "；", "：", "、", "，",
            ". ", "? ", "! ",
            " ",
            "",
        ],
        keep_separator=True,
    )

    document_items: List[KAGDocument] = []
    sentence_items: List[KAGDocument] = []
    for part_id, pack in doc2chunks.items():
        if not isinstance(part_id, str) or not part_id.strip():
            continue
        if not isinstance(pack, dict):
            continue
        chunks = pack.get("chunks") or []
        if not isinstance(chunks, list):
            continue
        base_md = dict(pack.get("document_metadata") or {})
        base_md["document_id"] = part_id
        for cidx, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("id", "") or "").strip() or f"{part_id}_chunk_{cidx}"
            chunk_text = str(chunk.get("content", "") or "").strip()
            if not chunk_text:
                continue
            chunk_md = dict(base_md)
            extra_md = chunk.get("metadata") or {}
            if isinstance(extra_md, dict):
                for key, value in extra_md.items():
                    if key not in chunk_md:
                        chunk_md[key] = value
            chunk_md["chunk_id"] = chunk_id
            chunk_md["vector_granularity"] = "document"
            chunk_md["parent_document_id"] = part_id
            document_items.append(KAGDocument(id=chunk_id, content=chunk_text, metadata=chunk_md))

            segments = splitter.split_text(chunk_text) if word_len(chunk_text, lang="auto") > sentence_chunk_size else [chunk_text]
            for sidx, seg in enumerate(segments, start=1):
                seg_text = str(seg or "").replace("\\n", "\n").strip()
                if not seg_text:
                    continue
                seg_md = dict(chunk_md)
                seg_md["vector_granularity"] = "sentence"
                seg_md["parent_document_id"] = chunk_id
                seg_md["sentence_order"] = sidx
                sentence_items.append(KAGDocument(id=f"{chunk_id}<->{sidx}", content=seg_text, metadata=seg_md))

    document_vector_store.store_documents(document_items)
    sentence_vector_store.store_documents(sentence_items)
    return {
        "document_count": len(document_items),
        "sentence_count": len(sentence_items),
    }


def _ensure_workspace_vector_stores_current(cfg: KAGConfig) -> None:
    workspace_dir = Path(str(cfg.knowledge_graph_builder.file_path)).parent
    kg_dir = workspace_dir / "knowledge_graph"
    vector_root = workspace_dir / "vector_store"
    vector_root.mkdir(parents=True, exist_ok=True)
    marker_path = vector_root / ".rebuilt_from_chunks.json"
    lock_path = vector_root / ".rebuilt_from_chunks.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            doc2chunks, source_path = _load_doc2chunks_for_vector_rebuild(kg_dir)
            source_mtime = float(source_path.stat().st_mtime)
            source_mtime_ns = int(source_path.stat().st_mtime_ns)

            marker_payload: Dict[str, Any] = {}
            if marker_path.exists():
                try:
                    loaded = json.loads(marker_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        marker_payload = loaded
                except Exception:
                    marker_payload = {}

            source_matches = (
                str(marker_payload.get("source_path", "")) == str(source_path)
                and int(marker_payload.get("source_mtime_ns", 0) or 0) == source_mtime_ns
            )
            stores_exist = (vector_root / "document").exists() and (vector_root / "sentence").exists()
            if source_matches and stores_exist:
                return

            stats = _persist_vector_stores_from_doc2chunks(cfg, doc2chunks)
            payload = {
                "source_path": str(source_path),
                "source_mtime": source_mtime,
                "source_mtime_ns": source_mtime_ns,
                "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **stats,
            }
            json_dump_atomic(str(marker_path), payload)
            logger.info(
                "Rebuilt vector stores from chunk artifacts: workspace=%s source=%s document=%d sentence=%d",
                workspace_dir,
                source_path.name,
                stats["document_count"],
                stats["sentence_count"],
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _json_list_length(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _runtime_graph_label_counts(cfg: KAGConfig) -> Dict[str, int]:
    raw_path = str(getattr(getattr(cfg, "storage", None), "graph_store_path", "") or "").strip()
    if not raw_path:
        return {}
    graph_path = Path(raw_path)
    if not graph_path.exists() or not graph_path.is_file():
        return {}
    try:
        with graph_path.open("rb") as f:
            graph_obj = pickle.load(f)
    except Exception:
        return {}
    if not hasattr(graph_obj, "nodes"):
        return {}
    counts: Dict[str, int] = {}
    try:
        for _, data in graph_obj.nodes(data=True):
            labels = data.get("type")
            if isinstance(labels, str):
                items = [labels]
            else:
                items = list(labels or [])
            for label in items:
                key = str(label or "").strip()
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def _interaction_sql_has_required_table(sql_db_path: Path) -> bool:
    if not sql_db_path.exists() or not sql_db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(str(sql_db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='Interaction_info'"
            )
            row = cur.fetchone()
            return bool(row and str(row[0] or "").strip() == "Interaction_info")
        finally:
            conn.close()
    except Exception:
        return False


def _existing_workspace_runtime_incomplete(cfg: KAGConfig) -> bool:
    workspace_dir = Path(str(cfg.knowledge_graph_builder.file_path)).parent
    narrative_global_dir = workspace_dir / "narrative_graph" / "global"
    episodes_json_path = narrative_global_dir / "episodes.json"
    storylines_json_path = narrative_global_dir / "storylines.json"
    interaction_paths = _article_interaction_artifact_paths(cfg)

    runtime_label_counts = _runtime_graph_label_counts(cfg)
    episode_json_count = _json_list_length(episodes_json_path)
    storyline_json_count = _json_list_length(storylines_json_path)
    episode_runtime_count = int(runtime_label_counts.get("Episode", 0) or 0)
    storyline_runtime_count = int(runtime_label_counts.get("Storyline", 0) or 0)

    if episode_json_count > 0 and episode_runtime_count <= 0:
        return True
    if storyline_json_count > 0 and storyline_runtime_count <= 0:
        return True
    if not _interaction_sql_has_required_table(interaction_paths["sql_db_path"]):
        return True
    return False


def _clear_graph(cfg: KAGConfig) -> None:
    store = GraphStore(cfg)
    try:
        store.reset_knowledge_graph()
    finally:
        store.close()


def _runtime_graph_has_data(cfg: KAGConfig) -> bool:
    raw_path = str(getattr(getattr(cfg, "storage", None), "graph_store_path", "") or "").strip()
    if not raw_path:
        return False
    graph_path = Path(raw_path)
    if not graph_path.exists() or not graph_path.is_file():
        return False
    try:
        if graph_path.stat().st_size <= 0:
            return False
    except Exception:
        return False
    try:
        with graph_path.open("rb") as f:
            graph_obj = pickle.load(f)
    except Exception:
        return False
    if not hasattr(graph_obj, "number_of_nodes") or not hasattr(graph_obj, "number_of_edges"):
        return False
    try:
        return int(graph_obj.number_of_nodes()) > 0 or int(graph_obj.number_of_edges()) > 0
    except Exception:
        return False


def _ensure_runtime_graph_embedding_cache(cfg: KAGConfig) -> bool:
    if not _runtime_graph_has_data(cfg):
        return False

    store = GraphStore(cfg)
    try:
        graph = store.get_graph()
        if graph.number_of_nodes() <= 0 and graph.number_of_edges() <= 0:
            return False
        doc_type = str(getattr(getattr(cfg, "global_config", None), "doc_type", "screenplay") or "screenplay")
        graph_query_utils = GraphQueryUtils(store, doc_type=doc_type)
        cache_status = graph_query_utils.get_embedding_cache_status()

        needs_sync = False
        if graph.number_of_nodes() > 0:
            node_emb_count = int(cache_status.get("graph_node_embedding_count", 0) or 0)
            entity_vec_count = int(cache_status.get("entity_vector_count", 0) or 0)
            if node_emb_count == 0 or node_emb_count > entity_vec_count:
                needs_sync = True
        if graph.number_of_edges() > 0:
            rel_emb_count = int(cache_status.get("graph_relation_embedding_count", 0) or 0)
            rel_vec_count = int(cache_status.get("relation_vector_count", 0) or 0)
            has_relation_descriptions = any(str(data.get("description") or "").strip() for _, _, _, data in graph.edges(keys=True, data=True))
            if rel_emb_count > rel_vec_count or (rel_emb_count == 0 and (rel_vec_count > 0 or has_relation_descriptions)):
                needs_sync = True

        if needs_sync:
            graph_query_utils.process_all_embeddings()
        return True
    finally:
        store.close()


def _load_existing_article_to_graph_store(cfg: KAGConfig) -> None:
    interaction_paths = _article_interaction_artifact_paths(cfg)
    kg_builder = KnowledgeGraphBuilder(cfg, use_memory=False)
    try:
        if interaction_paths["interaction_list_json_path"].exists():
            kg_builder.store_interactions_to_sql(
                interaction_list_json_path=str(interaction_paths["interaction_list_json_path"]),
                sql_db_path=str(interaction_paths["sql_db_path"]),
                reset_database=True,
                reset_table=True,
            )
        kg_builder.load_json_to_graph_store()
    finally:
        try:
            kg_builder.graph_store.close()
        except Exception:
            pass
    narrative_builder = NarrativeGraphBuilder(cfg)
    try:
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


def _build_article_workspace(cfg: KAGConfig, json_file_path: Path) -> Dict[str, Any]:
    interaction_paths = _article_interaction_artifact_paths(cfg)
    interaction_paths["interaction_dir"].mkdir(parents=True, exist_ok=True)
    interaction_paths["sql_db_path"].parent.mkdir(parents=True, exist_ok=True)
    builder = KnowledgeGraphBuilder(cfg, use_memory=False)
    try:
        builder.prepare_chunks(json_file_path=str(json_file_path), reset_output_dir=True, reset_vector_collections=True)
        builder.extract_entity_and_relation(
            retries=3,
            concurrency=min(16, int(cfg.knowledge_graph_builder.max_workers or 16)),
            per_task_timeout=int(cfg.knowledge_graph_builder.per_task_timeout or 2400),
            reset_outputs=True,
        )
        builder.run_extraction_refinement()
        builder.build_entity_and_relation_basic_info()
        builder.postprocess_and_save()
        builder.extract_properties()
        builder.extract_interactions(
            retries=3,
            concurrency=min(16, int(cfg.knowledge_graph_builder.max_workers or 16)),
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
            document_concurrency=min(16, int(cfg.narrative_graph_builder.max_workers or 16)),
            store_episode_support_edges=True,
            ensure_episode_embeddings=True,
            embedding_text_field="name_desc",
            embedding_batch_size=128,
        )
        narrative_builder.extract_episode_relations(
            episode_pair_concurrency=min(16, int(cfg.narrative_graph_builder.max_workers or 16)),
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
            similarity_threshold=0.5,
            overlap_pair_only=False,
            show_pair_progress=False,
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

    marker = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "json_file_path": str(json_file_path),
        "knowledge_graph_dir": cfg.knowledge_graph_builder.file_path,
        "narrative_graph_dir": cfg.narrative_graph_builder.file_path,
        "interaction_json_path": str(interaction_paths["interaction_json_path"]),
        "interaction_list_json_path": str(interaction_paths["interaction_list_json_path"]),
        "interaction_sql_db_path": str(interaction_paths["sql_db_path"]),
    }
    marker_path = Path(cfg.knowledge_graph_builder.file_path).parent / "build_marker.json"
    json_dump_atomic(str(marker_path), marker)
    return marker


def ensure_article_ready(
    *,
    base_cfg: KAGConfig,
    article_json_path: Path,
    workspace_dir: Path,
    runtime_library_path: Optional[Path] = None,
    runtime_tool_metadata_dir: Optional[Path] = None,
    strategy_enabled: bool = False,
    subagent_enabled: bool = False,
    rebuild: bool = False,
) -> KAGConfig:
    cfg = _build_article_config(
        base_cfg,
        workspace_dir=workspace_dir,
        runtime_library_path=runtime_library_path,
        runtime_tool_metadata_dir=runtime_tool_metadata_dir,
        strategy_enabled=strategy_enabled,
        subagent_enabled=subagent_enabled,
    )
    marker_path = workspace_dir / "build_marker.json"
    if marker_path.exists() and not rebuild and not _workspace_missing_artifacts(workspace_dir):
        if _runtime_graph_has_data(cfg) and not _existing_workspace_runtime_incomplete(cfg):
            _ensure_runtime_graph_embedding_cache(cfg)
            return cfg
        _load_existing_article_to_graph_store(cfg)
        return cfg
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _build_article_workspace(cfg, article_json_path)
    return cfg


def load_existing_article_workspace_or_raise(
    *,
    base_cfg: KAGConfig,
    workspace_dir: Path,
    runtime_library_path: Optional[Path] = None,
    runtime_tool_metadata_dir: Optional[Path] = None,
    strategy_enabled: bool = False,
    subagent_enabled: bool = False,
) -> KAGConfig:
    cfg = _build_article_config(
        base_cfg,
        workspace_dir=workspace_dir,
        runtime_library_path=runtime_library_path,
        runtime_tool_metadata_dir=runtime_tool_metadata_dir,
        strategy_enabled=strategy_enabled,
        subagent_enabled=subagent_enabled,
    )
    missing_artifacts = _workspace_missing_artifacts(workspace_dir)
    if missing_artifacts:
        raise FileNotFoundError(
            f"Existing article workspace not found or incomplete: {workspace_dir}. "
            "Refusing to rebuild because this run is configured to reuse extracted artifacts only. "
            f"Missing artifacts: {missing_artifacts}"
        )
    if _runtime_graph_has_data(cfg) and not _existing_workspace_runtime_incomplete(cfg):
        _ensure_runtime_graph_embedding_cache(cfg)
        return cfg
    _load_existing_article_to_graph_store(cfg)
    return cfg


def _write_offline_training_csv(out_path: Path, qa_rows: List[Dict[str, str]]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for row in qa_rows:
            writer.writerow(
                {
                    "question": _format_mcq_for_agent(str(row.get("question", "") or "")),
                    "answer": str(row.get("answer_text", "") or "").strip(),
                }
            )
    return out_path


def _write_online_training_csv(out_path: Path, qa_rows: List[Dict[str, str]]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question"])
        writer.writeheader()
        for row in qa_rows:
            writer.writerow({"question": _format_mcq_for_agent(str(row.get("question", "") or ""))})
    return out_path


def _build_online_batches(
    qa_rows: List[Dict[str, str]],
    *,
    warmup_questions: int = 0,
    batch_size: int = 3,
) -> List[List[Dict[str, str]]]:
    rows = list(qa_rows or [])
    if not rows:
        return []
    _ = warmup_questions
    step = max(1, int(batch_size or 1))
    batches: List[List[Dict[str, str]]] = []
    cursor = 0
    while cursor < len(rows):
        batches.append(rows[cursor : cursor + step])
        cursor += step
    return [batch for batch in batches if batch]


class StrategyLibraryAccumulator:
    def __init__(
        self,
        *,
        cfg: KAGConfig,
        runtime_library_path: Path,
        runtime_tool_metadata_dir: Path,
        output_dir: Path,
        dataset_name: str,
    ) -> None:
        self.cfg = copy.deepcopy(cfg)
        _set_global_language(self.cfg, "en")
        self.runtime_library_path = runtime_library_path
        self.runtime_tool_metadata_dir = runtime_tool_metadata_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        self.llm = OpenAILLM(self.cfg, llm_profile="memory")
        self.embedding_model = OpenAIEmbeddingModel(self.cfg.embedding)
        self.cluster_manager = self._build_cluster_manager()
        self.runtime_asset_manager = StrategyRuntimeAssetManager(
            library_path=str(self.runtime_library_path),
            tool_metadata_runtime_dir=str(self.runtime_tool_metadata_dir),
        )
        self.tool_candidates: List[Dict[str, Any]] = []
        self.template_counter = 0

    def _build_cluster_manager(self) -> StrategyTemplateClusterManager:
        return StrategyTemplateClusterManager(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            embedding_model=self.embedding_model,
            candidate_top_k=max(1, int(getattr(self.cfg.strategy_memory, "merge_candidate_top_k", 3) or 3)),
            min_candidate_score=float(getattr(self.cfg.strategy_memory, "merge_min_candidate_score", 0.28) or 0.28),
            consolidation_rounds=int(getattr(self.cfg.strategy_memory, "consolidation_rounds", 1) or 1),
            max_members_for_distill_prompt=int(getattr(self.cfg.strategy_memory, "cluster_distill_max_members", 12) or 12),
        )

    def _reset_memory_state(self) -> None:
        self.cluster_manager = self._build_cluster_manager()
        self.tool_candidates = []
        self.template_counter = 0

    def reset_state(self, *, clear_runtime: bool = False) -> None:
        self._reset_memory_state()
        if clear_runtime:
            self.clear_runtime()

    def clear_runtime(self) -> None:
        self.runtime_asset_manager.clear_all(aggregation_mode="narrative", dataset_name=self.dataset_name)

    def hydrate_from_runtime(self) -> Dict[str, int]:
        self._reset_memory_state()

        template_rows: List[Dict[str, Any]] = []
        if self.runtime_library_path.exists():
            try:
                payload = load_json(str(self.runtime_library_path))
                if isinstance(payload, dict) and isinstance(payload.get("templates"), list):
                    template_rows = [item for item in payload.get("templates", []) if isinstance(item, dict)]
            except Exception:
                template_rows = []

        seen_cluster_ids: set[str] = set()
        hydrated_templates = 0
        for idx, row in enumerate(template_rows, start=1):
            base_id = str(row.get("cluster_id", "") or row.get("template_id", "") or f"hydrated_template_{idx:05d}").strip()
            cluster_id = base_id
            suffix = 1
            while not cluster_id or cluster_id in seen_cluster_ids:
                cluster_id = f"{base_id or 'hydrated_template'}__{suffix}"
                suffix += 1
            seen_cluster_ids.add(cluster_id)
            sanitized = self.cluster_manager.sanitize_template(dict(row), default_id=cluster_id)
            self.cluster_manager.raw_templates.append(sanitized)
            cluster = {
                "cluster_id": cluster_id,
                "created_at": str(row.get("created_at", "") or time.strftime("%Y-%m-%d %H:%M:%S")),
                "member_templates": [sanitized],
                "pattern_name": sanitized.get("pattern_name", "Generic Retrieval Pattern"),
                "pattern_description": sanitized.get("pattern_description", ""),
                "recommended_chain": sanitized.get("recommended_chain", []),
                "anti_patterns": sanitized.get("anti_patterns", []),
                "query_pattern_prototype": sanitized.get("query_pattern") if isinstance(sanitized.get("query_pattern"), dict) else {},
                "query_abstract": sanitized.get("query_abstract", ""),
                "query_pattern_text": sanitized.get("query_pattern_text", ""),
                "pattern_embedding": sanitized.get("pattern_embedding"),
            }
            cluster = self.cluster_manager._recompute_cluster_summary_with_mode(cluster, use_llm=False)
            self.cluster_manager.clusters.append(cluster)
            hydrated_templates += 1

        hydrated_tool_overrides = 0
        for language in ("zh", "en"):
            override_path = self.runtime_tool_metadata_dir / language / "strategy_runtime_overrides.json"
            if not override_path.exists():
                continue
            try:
                payload = load_json(str(override_path))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for tool_name, item in payload.items():
                if not isinstance(item, dict):
                    continue
                resolved_tool_name = str(item.get("name", "") or tool_name or "").strip()
                description = str(item.get("description", "") or "").strip()
                if not resolved_tool_name or not description:
                    continue
                self.tool_candidates.append(
                    {
                        "tool_name": resolved_tool_name,
                        "language": language,
                        "current_description": description,
                        "decision": "revise",
                        "proposed_description": description,
                    }
                )
                hydrated_tool_overrides += 1

        self.template_counter = hydrated_templates
        return {
            "template_count": hydrated_templates,
            "tool_override_count": hydrated_tool_overrides,
        }

    def add_tool_candidates(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows or []:
            if isinstance(row, dict):
                self.tool_candidates.append(dict(row))

    def add_templates(self, rows: List[Dict[str, Any]], *, template_prefix: str) -> int:
        prepared: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            self.template_counter += 1
            original_id = str(item.get("template_id", "") or f"tmp_{self.template_counter:05d}")
            item["template_id"] = f"{template_prefix}__{original_id}"
            prepared.append(item)
        self.cluster_manager.add_templates(prepared)
        return len(prepared)

    def _aggregate_tool_candidates(self) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self.tool_candidates:
            tool_name = str(row.get("tool_name", "") or "").strip()
            language = str(row.get("language", "") or "").strip().lower()
            if not tool_name or language not in {"zh", "en"}:
                continue
            key = (tool_name, language)
            current = grouped.get(key)
            if current is None:
                current = {
                    "tool_name": tool_name,
                    "language": language,
                    "current_description": str(row.get("current_description", "") or ""),
                    "decision_votes": {},
                    "proposal_votes": {},
                }
                grouped[key] = current
            decision = str(row.get("decision", "") or "").strip().lower()
            proposal = str(row.get("proposed_description", "") or "").strip()
            current["decision_votes"][decision] = int(current["decision_votes"].get(decision, 0)) + 1
            if proposal:
                current["proposal_votes"][proposal] = int(current["proposal_votes"].get(proposal, 0)) + 1

        out: List[Dict[str, Any]] = []
        for _, row in sorted(grouped.items()):
            decision_votes = row["decision_votes"]
            proposal_votes = row["proposal_votes"]
            revise_count = int(decision_votes.get("revise", 0))
            keep_count = int(decision_votes.get("keep", 0))
            if revise_count > keep_count and proposal_votes:
                proposed_description = sorted(proposal_votes.items(), key=lambda item: (item[1], len(item[0])), reverse=True)[0][0]
                decision = "revise"
            else:
                proposed_description = str(row.get("current_description", "") or "")
                decision = "keep"
            out.append(
                {
                    "tool_name": row["tool_name"],
                    "language": row["language"],
                    "current_description": str(row.get("current_description", "") or ""),
                    "decision": decision,
                    "proposed_description": proposed_description,
                    "support_count": max(revise_count, keep_count),
                    "keep_count": keep_count,
                    "revise_count": revise_count,
                }
            )
        return out

    def export(self, *, consolidate: bool, export_name: str) -> Dict[str, Any]:
        if consolidate:
            try:
                self.cluster_manager.consolidate()
            except Exception:
                pass
        cluster_payload = self.cluster_manager.export_training_payload()
        runtime_templates = self.cluster_manager.runtime_templates()
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        training_library_payload = {
            "library_version": 2,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": generated_at,
            "raw_template_count": len(self.cluster_manager.raw_templates),
            "cluster_count": int(cluster_payload.get("cluster_count", 0) or 0),
            "clusters": cluster_payload.get("clusters", []),
            "templates": runtime_templates,
        }
        runtime_library_payload = {
            "library_version": 2,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": generated_at,
            "template_count": len(runtime_templates),
            "templates": runtime_templates,
        }
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
        tool_candidates = self._aggregate_tool_candidates()

        out_dir = self.output_dir / export_name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_dump_atomic(str(out_dir / "strategy_library_training.json"), training_library_payload)
        json_dump_atomic(str(out_dir / "template_source_index.json"), template_source_index)
        json_dump_atomic(str(out_dir / "tool_description_candidates.json"), tool_candidates)
        json_dump_atomic(str(self.runtime_library_path), runtime_library_payload)
        runtime_tool_paths = self.runtime_asset_manager.export_tool_metadata_overrides(tool_candidates)
        runtime_source_index_path = self.runtime_asset_manager.export_template_source_index(template_source_index)
        return {
            "training_library_path": str(out_dir / "strategy_library_training.json"),
            "runtime_library_path": str(self.runtime_library_path),
            "cluster_count": int(cluster_payload.get("cluster_count", 0) or 0),
            "template_count": len(runtime_templates),
            "template_source_index_path": str(out_dir / "template_source_index.json"),
            "tool_description_candidate_path": str(out_dir / "tool_description_candidates.json"),
            "runtime_tool_metadata_paths": runtime_tool_paths,
            "runtime_template_source_index_path": runtime_source_index_path,
        }


@dataclass
class OnlineRepeatState:
    setting_name: str
    repeat_index: int
    runtime_root: Path
    runtime_library_path: Path
    runtime_tool_metadata_dir: Path
    checkpoint_root: Path
    accumulator: StrategyLibraryAccumulator
    last_checkpoint_article: str = ""

    def checkpoint_dir(self, article_name: str) -> Path:
        return self.checkpoint_root / article_name


class AgentThreadLocal:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, enable_sql_tools: bool) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.enable_sql_tools = bool(enable_sql_tools)
        base_hidden_tool_names: set[str] = {
            "retrieve_entity_by_id",
            "get_common_neighbors",
            "find_paths_between_nodes",
            "get_co_section_entities",
            "get_k_hop_subgraph",
            "vdb_search_docs",
            "vdb_search_hierdocs",
            "search_related_content",
        }
        extra_hidden_tool_names = {
            str(name or "").strip()
            for name in (getattr(getattr(cfg, "strategy_memory", None), "hidden_tool_names", []) or [])
            if str(name or "").strip()
        }
        self.hidden_tool_names: set[str] = base_hidden_tool_names | extra_hidden_tool_names
        self.local = threading.local()
        self.choice_extractor = ChoiceExtractor(cfg)
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _build_state(self) -> Dict[str, Any]:
        agent = QuestionAnsweringAgent(
            self.cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
        )
        setattr(agent, "hidden_tool_names", set(self.hidden_tool_names))
        finalizer_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        state = {"agent": agent, "finalizer_llm": finalizer_llm}
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


class AgentEvaluator:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, article_name: str, enable_sql_tools: bool) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.article_name = article_name
        self.open_answer_choice_adapter_enabled = False
        self.option_disambiguation_enabled = False
        self.mcq_answer_adapter_enabled = True
        self.terminal_mcq_enforcement_enabled = True
        self.posthoc_choice_recovery_enabled = True
        self.tlocal = AgentThreadLocal(cfg, setting_name=setting_name, enable_sql_tools=enable_sql_tools)
        self.choice_extractor = ChoiceExtractor(cfg)
        self.hybrid_tlocal = TraditionalHybridThreadLocal(cfg) if setting_name == "no_strategy_agent" else None

    def _adapt_open_answer_to_choice(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        current_answer: str,
    ) -> str:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return current_answer
        direct_choice = self.choice_extractor.choose(
            question_text=str(row.get("question", "") or ""),
            answer_text=current_answer,
        )
        if direct_choice:
            return _ensure_answer_confidence(current_answer)
        stem = str(parsed.get("question_stem", "") or "").strip()
        agent_answer_text = _extract_semantic_answer_text(current_answer)
        if not stem or not agent_answer_text:
            return current_answer
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        prompt = "\n".join(
            [
                "You are mapping an open-ended answer to one answer choice.",
                "Return JSON only:",
                '{"answer_choice":"A","answer_text":"...","evidence":"mapped_from_agent_answer","confidence":0.72}',
                "",
                "Rules:",
                "- Use the agent answer as the primary signal.",
                "- Compare the agent answer against all options and choose the single closest supported option.",
                "- Do not use outside knowledge.",
                "- `answer_text` must restate the chosen option text.",
                "- `evidence` should briefly explain the semantic match from the agent answer.",
                "- `confidence` must be between 0 and 1.",
                "- If the agent answer is underspecified or partially matches multiple options, still choose the closest option and lower confidence.",
                "",
                "Question:",
                stem,
                "",
                "Choices:",
                choices_block,
                "",
                "Agent answer:",
                _trim_text(agent_answer_text, limit=700),
            ]
        )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            adapted = _extract_llm_text(result)
            if self.choice_extractor.choose(
                question_text=str(row.get("question", "") or ""),
                answer_text=adapted,
            ):
                return _ensure_answer_confidence(adapted)
        except Exception:
            return current_answer
        return current_answer

    def _adapt_mcq_answer_to_choice(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return current_answer
        direct_choice = self.choice_extractor.choose(
            question_text=str(row.get("question", "") or ""),
            answer_text=current_answer,
        )
        if direct_choice and not _looks_like_tool_call_payload(current_answer):
            return _ensure_answer_confidence(current_answer)
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses, max_items=8)
        if not evidence_block:
            return current_answer
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        prompt = "\n".join(
            [
                "You are the final multiple-choice answer adapter.",
                "Your job is to convert retrieved evidence and any incomplete prior answer into one valid final choice.",
                "Return JSON only:",
                '{"answer_choice":"A","answer_text":"...","evidence":"...","confidence":0.72}',
                "",
                "Rules:",
                "- Choose exactly one option label from the provided choices.",
                "- Use the retrieved evidence as the primary signal.",
                "- `answer_text` must restate the chosen option text.",
                "- `evidence` must be brief and grounded in the retrieved evidence.",
                "- `confidence` must be a number between 0 and 1.",
                "- Compare all options before choosing.",
                "- If the previous answer is just a tool call, an unfinished intermediate result, or a non-choice explanation, ignore its format and still produce a final answer.",
                "- If the evidence is weak, choose the least unsupported option and lower confidence rather than returning no choice.",
                "- Do not use outside knowledge.",
                "",
                "Question:",
                str(parsed.get("question_stem", "") or ""),
                "",
                "Choices:",
                choices_block,
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
            if self.choice_extractor.choose(
                question_text=str(row.get("question", "") or ""),
                answer_text=adapted,
            ):
                return _ensure_answer_confidence(adapted)
        except Exception:
            return current_answer
        return current_answer

    def _repair_final_answer(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        if not tool_uses:
            return current_answer
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return current_answer
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses)
        if not evidence_block:
            return current_answer
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        prompt = "\n".join(
            [
                "You are given a multiple-choice question and retrieved evidence from tools.",
                "Choose the single best answer supported by the evidence.",
                "Return JSON only:",
                '{"answer_choice":"A","answer_text":"...","evidence":"...","confidence":0.72}',
                "",
                "Rules:",
                "- Use only the retrieved evidence below.",
                "- Choose exactly one option label from the provided choices.",
                "- `answer_text` must restate the chosen option text.",
                "- `evidence` must be brief and grounded.",
                "- `confidence` must be a number between 0 and 1 reflecting how well the retrieved evidence supports the chosen option.",
                "- Compare all options against the evidence before deciding.",
                "- Prefer the option with the strongest direct support and the fewest unsupported assumptions.",
                "- If an option adds a motive, warning, implication, or relationship that is not supported by the evidence, reject that option.",
                "- Treat missing evidence as a reason not to choose an option.",
                "- Do not preserve the previous answer unless it is still the best-supported option.",
                "",
                "Question:",
                str(parsed.get("question_stem", "") or ""),
                "",
                "Choices:",
                choices_block,
                "",
                "Retrieved evidence:",
                evidence_block,
                "",
                "Previous incomplete answer:",
                _trim_text(current_answer, limit=600) or "(none)",
            ]
        )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            repaired = _extract_llm_text(result)
            if _extract_choice_label_heuristic(repaired, order):
                return _ensure_answer_confidence(repaired)
        except Exception:
            return current_answer
        return current_answer

    def _build_disambiguation_query(self, *, row: Dict[str, str]) -> str:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        stem = str(parsed.get("question_stem", "") or "").strip()
        if not stem or not order or not choices:
            return ""
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        if not choices_block:
            return ""
        return "\n".join(
            [
                stem,
                "Choices:",
                choices_block,
                "Focus on evidence that distinguishes the competing options.",
            ]
        ).strip()

    def _run_low_confidence_scene_narrative_fallback(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if self.setting_name != "no_strategy_agent":
            return []
        disambiguation_query = self._build_disambiguation_query(row=row)
        if not disambiguation_query:
            return []
        extra_tool_uses: List[Dict[str, Any]] = []
        existing_names = self._normalize_tool_name_set(tool_uses or [])
        lowered_question = str(row.get("question", "") or "").lower()
        need_narrative = any(
            cue in lowered_question
            for cue in (
                "why",
                "imply",
                "most likely",
                "warning",
                "attitude",
                "feel",
                "suggest",
                "dilemma",
                "motive",
            )
        )
        if "section_evidence_search" not in existing_names:
            section_hit = self._call_extra_retrieval_tool(
                state=state,
                tool_name="section_evidence_search",
                arguments={
                    "query": disambiguation_query,
                    "section_top_k": 6,
                    "max_length": 240,
                    "related_entity_limit": 2,
                },
            )
            if section_hit:
                extra_tool_uses.append(section_hit)
        if "bm25_search_docs" not in existing_names:
            bm25_hit = self._call_extra_retrieval_tool(
                state=state,
                tool_name="bm25_search_docs",
                arguments={
                    "query": disambiguation_query,
                    "k": 8,
                },
            )
            if bm25_hit:
                extra_tool_uses.append(bm25_hit)
        if "vdb_search_sentences" not in existing_names:
            sentence_hit = self._call_extra_retrieval_tool(
                state=state,
                tool_name="vdb_search_sentences",
                arguments={
                    "query": disambiguation_query,
                    "limit": 8,
                },
            )
            if sentence_hit:
                extra_tool_uses.append(sentence_hit)
        if need_narrative and not (existing_names & {"narrative_hierarchical_search", "entity_event_trace_search", "fact_timeline_resolution_search"}):
            narrative_hit = self._call_extra_retrieval_tool(
                state=state,
                tool_name="narrative_hierarchical_search",
                arguments={
                    "query": disambiguation_query,
                    "storyline_top_k": 3,
                    "episode_top_k": 4,
                    "event_top_k": 6,
                    "document_top_k": 4,
                    "max_evidence_length": 220,
                },
            )
            if narrative_hit:
                extra_tool_uses.append(narrative_hit)
        return extra_tool_uses

    @staticmethod
    def _normalize_tool_name_set(tool_uses: List[Dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for item in tool_uses or []:
            raw_name = str((item or {}).get("tool_name", "") or "").strip()
            if not raw_name:
                continue
            normalized = raw_name
            if normalized.startswith("evaluator_"):
                normalized = normalized[len("evaluator_") :]
            names.add(normalized)
        return names

    def _has_strong_agent_evidence(self, tool_uses: List[Dict[str, Any]]) -> bool:
        names = self._normalize_tool_name_set(tool_uses)
        local_hits = names & {
            "section_evidence_search",
            "bm25_search_docs",
            "vdb_search_sentences",
            "choice_grounded_evidence_search",
            "fact_timeline_resolution_search",
        }
        narrative_hits = names & {
            "narrative_hierarchical_search",
            "entity_event_trace_search",
            "fact_timeline_resolution_search",
        }
        return (len(local_hits) >= 2) or (bool(local_hits) and bool(narrative_hits))

    def _needs_option_disambiguation_support(
        self,
        *,
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
    ) -> bool:
        if self.setting_name != "no_strategy_agent":
            return False
        if self._has_strong_agent_evidence(tool_uses):
            return False
        names = self._normalize_tool_name_set(tool_uses)
        if "choice_grounded_evidence_search" in names:
            return False
        question_text = str(row.get("question", "") or "")
        return bool(_parse_choices(question_text).get("choice_order"))

    def _get_tool_by_name(self, *, state: Dict[str, Any], tool_name: str) -> Any:
        agent = state.get("agent")
        if agent is None:
            return None
        getter = getattr(agent, "_all_tools", None)
        if not callable(getter):
            return None
        try:
            tools = list(getter() or [])
        except Exception:
            return None
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name == tool_name:
                return tool
        return None

    def _call_extra_retrieval_tool(
        self,
        *,
        state: Dict[str, Any],
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        tool = self._get_tool_by_name(state=state, tool_name=tool_name)
        if tool is None:
            return None
        try:
            raw_output = tool.call(json.dumps(arguments, ensure_ascii=False))
        except Exception:
            return None
        output_text = _trim_text(raw_output, limit=1800)
        if not output_text:
            return None
        return {
            "tool_name": f"evaluator_{tool_name}",
            "tool_arguments": json.dumps(arguments, ensure_ascii=False),
            "tool_output": output_text,
        }

    def _augment_tool_uses_for_option_disambiguation(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        if (not force) and (not self._needs_option_disambiguation_support(row=row, tool_uses=tool_uses)):
            return []
        extra_tool_uses: List[Dict[str, Any]] = []
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return extra_tool_uses
        disambiguation_query = self._build_disambiguation_query(row=row)
        if not disambiguation_query:
            return extra_tool_uses
        existing_names = self._normalize_tool_name_set(tool_uses)
        section_hit = self._call_extra_retrieval_tool(
            state=state,
            tool_name="section_evidence_search",
            arguments={
                "query": disambiguation_query,
                "section_top_k": 6,
                "max_length": 240,
                "related_entity_limit": 2,
            },
        )
        if section_hit and "section_evidence_search" not in existing_names:
            extra_tool_uses.append(section_hit)
        bm25_hit = self._call_extra_retrieval_tool(
            state=state,
            tool_name="bm25_search_docs",
            arguments={
                "query": disambiguation_query,
                "k": 6,
            },
        )
        if bm25_hit and "bm25_search_docs" not in existing_names:
            extra_tool_uses.append(bm25_hit)
        lowered_question = str(row.get("question", "") or "").lower()
        if any(cue in lowered_question for cue in ("why", "imply", "most likely", "feel", "attitude", "warning")):
            narrative_hit = self._call_extra_retrieval_tool(
                state=state,
                tool_name="narrative_hierarchical_search",
                arguments={
                    "query": disambiguation_query,
                    "storyline_top_k": 3,
                    "episode_top_k": 4,
                    "event_top_k": 5,
                    "document_top_k": 4,
                    "max_evidence_length": 220,
                },
            )
            if narrative_hit and "narrative_hierarchical_search" not in existing_names:
                extra_tool_uses.append(narrative_hit)
        return extra_tool_uses

    def _force_terminal_mcq_choice(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return current_answer
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses, max_items=8)
        prompt = "\n".join(
            [
                "You are the terminal multiple-choice answer finalizer.",
                "You must output exactly one valid answer choice.",
                "Return JSON only:",
                '{"answer_choice":"A","answer_text":"...","evidence":"...","confidence":0.51}',
                "",
                "Rules:",
                "- You must choose exactly one option label from the provided choices.",
                "- Never return a tool call, intermediate plan, or abstention.",
                "- Use the retrieved evidence as the primary signal.",
                "- If the evidence is incomplete, choose the least unsupported option and lower confidence.",
                "- `answer_text` must restate the chosen option text.",
                "- `evidence` must be brief and grounded in the retrieved evidence or explain the closest-supported match.",
                "- `confidence` must be a number between 0 and 1.",
                "- Do not use outside knowledge.",
                "",
                "Question:",
                str(parsed.get("question_stem", "") or ""),
                "",
                "Choices:",
                choices_block,
                "",
                "Retrieved evidence:",
                evidence_block or "(none)",
                "",
                "Previous answer or unfinished draft:",
                _trim_text(current_answer, limit=700) or "(none)",
            ]
        )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            forced = _extract_llm_text(result)
            if self.choice_extractor.choose(
                question_text=str(row.get("question", "") or ""),
                answer_text=forced,
            ):
                return _ensure_answer_confidence(forced, default=0.51)
        except Exception:
            return current_answer
        return current_answer

    def _force_terminal_mcq_label_only(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = [str(x).strip().upper() for x in (parsed.get("choice_order") or []) if str(x).strip()]
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return ""
        choices_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in order
            if str(choices.get(label, "") or "").strip()
        )
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses, max_items=8)
        prompt = "\n".join(
            [
                "Choose the single best answer choice.",
                "Output exactly one uppercase letter and nothing else.",
                f"Valid labels: {', '.join(order)}",
                "",
                "Question:",
                str(parsed.get("question_stem", "") or ""),
                "",
                "Choices:",
                choices_block,
                "",
                "Retrieved evidence:",
                evidence_block or "(none)",
                "",
                "Previous answer:",
                _trim_text(current_answer, limit=500) or "(none)",
            ]
        )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            raw = _extract_llm_text(result)
        except Exception:
            return ""
        label = _extract_choice_label_heuristic(raw, order)
        if label:
            return label
        stripped = str(raw or "").strip().upper()
        return stripped if stripped in set(order) else ""

    def _enforce_terminal_mcq_answer(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
        current_choice: str,
        current_confidence: Optional[float],
    ) -> tuple[str, str, Optional[float], List[Dict[str, Any]], bool]:
        parsed = _parse_choices(str(row.get("question", "") or ""))
        if not parsed.get("choice_order"):
            return current_answer, current_choice, current_confidence, tool_uses, False
        predicted_choice = str(current_choice or "").strip().upper()
        final_answer = str(current_answer or "")
        final_confidence = current_confidence
        merged_tool_uses = list(tool_uses or [])
        used = False

        must_recover = (not predicted_choice) or _looks_like_tool_call_payload(final_answer)
        if must_recover:
            targeted_tool_uses = self._augment_tool_uses_for_option_disambiguation(
                state=state,
                row=row,
                tool_uses=merged_tool_uses,
                force=True,
            )
            if targeted_tool_uses:
                merged_tool_uses = list(targeted_tool_uses) + list(merged_tool_uses)
                used = True

            choice_tool_hit, suggested_choice = self._run_mcq_choice_tool_fallback(
                state=state,
                row=row,
                tool_uses=merged_tool_uses,
            )
            if choice_tool_hit:
                merged_tool_uses = [choice_tool_hit, *merged_tool_uses]
                used = True
                if suggested_choice:
                    choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
                    choice_text = str((choices or {}).get(suggested_choice, "") or "").strip()
                    final_answer = json.dumps(
                        {
                            "answer_choice": suggested_choice,
                            "answer_text": choice_text,
                            "evidence": "choice_grounded_evidence_search ranked this option highest under option-level evidence comparison.",
                            "confidence": 0.64,
                        },
                        ensure_ascii=False,
                    )
                    predicted_choice = suggested_choice
                    final_confidence = 0.64

            if (not predicted_choice) or _looks_like_tool_call_payload(final_answer):
                adapted_answer = self._adapt_mcq_answer_to_choice(
                    state=state,
                    row=row,
                    tool_uses=merged_tool_uses,
                    current_answer=final_answer,
                )
                adapted_choice = self.choice_extractor.choose(
                    question_text=str(row.get("question", "") or ""),
                    answer_text=adapted_answer,
                )
                if adapted_choice:
                    final_answer = _ensure_answer_confidence(adapted_answer)
                    predicted_choice = adapted_choice
                    final_confidence = _extract_answer_confidence(final_answer)
                    used = True

            if not predicted_choice:
                forced_answer = self._force_terminal_mcq_choice(
                    state=state,
                    row=row,
                    tool_uses=merged_tool_uses,
                    current_answer=final_answer,
                )
                forced_choice = self.choice_extractor.choose(
                    question_text=str(row.get("question", "") or ""),
                    answer_text=forced_answer,
                )
                if forced_choice:
                    final_answer = _ensure_answer_confidence(forced_answer, default=0.51)
                    predicted_choice = forced_choice
                    final_confidence = _extract_answer_confidence(final_answer)
                    used = True

            if not predicted_choice:
                forced_label = self._force_terminal_mcq_label_only(
                    state=state,
                    row=row,
                    tool_uses=merged_tool_uses,
                    current_answer=final_answer,
                )
                if forced_label:
                    choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
                    choice_text = str((choices or {}).get(forced_label, "") or "").strip()
                    final_answer = json.dumps(
                        {
                            "answer_choice": forced_label,
                            "answer_text": choice_text,
                            "evidence": "terminal_label_only_forced_choice",
                            "confidence": 0.34,
                        },
                        ensure_ascii=False,
                    )
                    predicted_choice = forced_label
                    final_confidence = 0.34
                    used = True

        return final_answer, predicted_choice, final_confidence, merged_tool_uses, used

    def _run_mcq_choice_tool_fallback(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, str],
        tool_uses: List[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if self.setting_name != "no_strategy_agent":
            return None, ""
        parsed = _parse_choices(str(row.get("question", "") or ""))
        order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not order or not choices:
            return None, ""
        existing_names = self._normalize_tool_name_set(tool_uses)
        if "choice_grounded_evidence_search" in existing_names:
            return None, ""
        choice_hit = self._call_extra_retrieval_tool(
            state=state,
            tool_name="choice_grounded_evidence_search",
            arguments={
                "query": str(row.get("question", "") or "").strip(),
                "section_top_k": 3,
                "document_top_k": 2,
                "sentence_top_k": 3,
                "max_length": 180,
                "use_llm_judge": False,
            },
        )
        if not choice_hit:
            return None, ""
        suggested_choice = _extract_choice_from_choice_tool_output(
            str(choice_hit.get("tool_output", "") or ""),
            order,
        )
        return choice_hit, suggested_choice

    def _should_calibrate_final_answer(self, *, question_text: str, tool_uses: List[Dict[str, Any]]) -> bool:
        if not tool_uses:
            return False
        if self.setting_name == "no_strategy_agent":
            return True
        lowered = str(question_text or "").lower()
        inferential_cues = (
            "infer",
            "inference",
            "imply",
            "implied",
            "most likely",
            "likely",
            "why",
            "warning",
            "conclude",
            "conclusion",
            "feel",
            "attitude",
            "allusion",
            "suggest",
            "purpose",
            "mission",
        )
        return any(cue in lowered for cue in inferential_cues)

    def evaluate_row(self, row: Dict[str, str], run_index: int) -> EvalResult:
        started = time.time()
        state = self.tlocal.state()
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        raw_agent_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        strategy_context: Dict[str, Any] = {}
        error_text = ""
        final_answer_confidence: Optional[float] = None
        low_confidence_fallback_used = False
        mcq_answer_adapter_used = False
        terminal_mcq_enforcement_used = False
        posthoc_choice_recovery_used = False
        if self.open_answer_choice_adapter_enabled:
            base_prompt = _format_open_question_for_agent(str(row.get("question", "") or ""))
            guarded_prompt = _format_open_question_for_agent_with_retrieval_guard(str(row.get("question", "") or ""))
        else:
            base_prompt = _format_mcq_for_agent(str(row.get("question", "") or ""))
            guarded_prompt = _format_mcq_for_agent_with_retrieval_guard(str(row.get("question", "") or ""))
        prompt_variants = [
            ("base", base_prompt, (10, 6, 4)),
            ("guarded", guarded_prompt, (12, 8, 6)),
        ]
        for prompt_tag, prompt, max_calls_seq in prompt_variants:
            for max_calls in max_calls_seq:
                try:
                    responses = state["agent"].ask(
                        prompt,
                        lang="en",
                        session_id=f"{self.setting_name}_{self.article_name}_{row['question_id']}_{run_index}_{prompt_tag}_{max_calls}",
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
        predicted_choice = ""
        if not error_text and tool_uses:
            if self.open_answer_choice_adapter_enabled:
                adapted_answer = self._adapt_open_answer_to_choice(
                    state=state,
                    row=row,
                    current_answer=final_answer,
                )
                adapted_choice = self.choice_extractor.choose(
                    question_text=str(row.get("question", "") or ""),
                    answer_text=adapted_answer,
                )
                if adapted_choice:
                    final_answer = _ensure_answer_confidence(adapted_answer)
                    predicted_choice = adapted_choice
                    final_answer_confidence = _extract_answer_confidence(final_answer)
            else:
                if self.option_disambiguation_enabled:
                    extra_tool_uses = self._augment_tool_uses_for_option_disambiguation(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                    )
                    if extra_tool_uses:
                        tool_uses = list(extra_tool_uses) + list(tool_uses)
                if self._should_calibrate_final_answer(
                    question_text=str(row.get("question", "") or ""),
                    tool_uses=tool_uses,
                ):
                    calibrated_answer = self._repair_final_answer(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                        current_answer=final_answer,
                    )
                    calibrated_choice = self.choice_extractor.choose(
                        question_text=str(row.get("question", "") or ""),
                        answer_text=calibrated_answer,
                    )
                    if calibrated_choice:
                        final_answer = _ensure_answer_confidence(calibrated_answer)
                        predicted_choice = calibrated_choice
                        final_answer_confidence = _extract_answer_confidence(final_answer)
                predicted_choice = self.choice_extractor.choose(
                    question_text=str(row.get("question", "") or ""),
                    answer_text=final_answer,
                )
                if not predicted_choice:
                    repaired_choice = ""
                    repaired_answer = self._repair_final_answer(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                        current_answer=final_answer,
                    )
                    repaired_choice = self.choice_extractor.choose(
                        question_text=str(row.get("question", "") or ""),
                        answer_text=repaired_answer,
                    )
                    if repaired_choice:
                        final_answer = _ensure_answer_confidence(repaired_answer)
                        predicted_choice = repaired_choice
                        final_answer_confidence = _extract_answer_confidence(final_answer)
                if self.option_disambiguation_enabled and ((not predicted_choice) or _is_low_confidence(final_answer_confidence)):
                    choice_tool_hit, suggested_choice = self._run_mcq_choice_tool_fallback(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                    )
                    if choice_tool_hit:
                        tool_uses = [choice_tool_hit, *tool_uses]
                        if suggested_choice:
                            parsed = _parse_choices(str(row.get("question", "") or ""))
                            choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
                            choice_text = str((choices or {}).get(suggested_choice, "") or "").strip()
                            final_answer = json.dumps(
                                {
                                    "answer_choice": suggested_choice,
                                    "answer_text": choice_text,
                                    "evidence": "choice_grounded_evidence_search ranked this option highest under option-level evidence comparison.",
                                    "confidence": 0.64,
                                },
                                ensure_ascii=False,
                            )
                            predicted_choice = suggested_choice
                            final_answer_confidence = 0.64
                        else:
                            repaired_answer = self._repair_final_answer(
                                state=state,
                                row=row,
                                tool_uses=tool_uses,
                                current_answer=final_answer,
                            )
                            repaired_choice = self.choice_extractor.choose(
                                question_text=str(row.get("question", "") or ""),
                                answer_text=repaired_answer,
                            )
                            if repaired_choice:
                                final_answer = _ensure_answer_confidence(repaired_answer)
                                predicted_choice = repaired_choice
                                final_answer_confidence = _extract_answer_confidence(final_answer)
                if self.mcq_answer_adapter_enabled and (
                    (not predicted_choice)
                    or _looks_like_tool_call_payload(final_answer)
                    or _is_low_confidence(final_answer_confidence)
                ):
                    adapted_answer = self._adapt_mcq_answer_to_choice(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                        current_answer=final_answer,
                    )
                    adapted_choice = self.choice_extractor.choose(
                        question_text=str(row.get("question", "") or ""),
                        answer_text=adapted_answer,
                    )
                    if adapted_choice:
                        final_answer = _ensure_answer_confidence(adapted_answer)
                        predicted_choice = adapted_choice
                        final_answer_confidence = _extract_answer_confidence(final_answer)
                        mcq_answer_adapter_used = True
                elif final_answer_confidence is None:
                    final_answer_confidence = _extract_answer_confidence(final_answer)
                if predicted_choice and not low_confidence_fallback_used and _is_low_confidence(final_answer_confidence):
                    fallback_tool_uses = self._run_low_confidence_scene_narrative_fallback(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                    )
                    if fallback_tool_uses:
                        fallback_answer = self._repair_final_answer(
                            state=state,
                            row=row,
                            tool_uses=list(fallback_tool_uses) + list(tool_uses),
                            current_answer=final_answer,
                        )
                        fallback_choice = self.choice_extractor.choose(
                            question_text=str(row.get("question", "") or ""),
                            answer_text=fallback_answer,
                        )
                        if fallback_choice:
                            final_answer = _ensure_answer_confidence(fallback_answer)
                            predicted_choice = fallback_choice
                            final_answer_confidence = _extract_answer_confidence(final_answer)
                            tool_uses = list(fallback_tool_uses) + list(tool_uses)
                            low_confidence_fallback_used = True
                if predicted_choice and final_answer_confidence is None:
                    final_answer = _ensure_answer_confidence(final_answer)
                    final_answer_confidence = _extract_answer_confidence(final_answer)
                if self.terminal_mcq_enforcement_enabled:
                    final_answer, predicted_choice, final_answer_confidence, tool_uses, terminal_used = self._enforce_terminal_mcq_answer(
                        state=state,
                        row=row,
                        tool_uses=tool_uses,
                        current_answer=final_answer,
                        current_choice=predicted_choice,
                        current_confidence=final_answer_confidence,
                    )
                    terminal_mcq_enforcement_used = terminal_mcq_enforcement_used or terminal_used
        if self.posthoc_choice_recovery_enabled and not predicted_choice and final_answer:
            recovered_choice = self.choice_extractor.choose(
                question_text=str(row.get("question", "") or ""),
                answer_text=final_answer,
            )
            if recovered_choice:
                predicted_choice = recovered_choice
                posthoc_choice_recovery_used = True
        if self.posthoc_choice_recovery_enabled and not predicted_choice and raw_agent_answer and raw_agent_answer != final_answer:
            recovered_choice = self.choice_extractor.choose(
                question_text=str(row.get("question", "") or ""),
                answer_text=raw_agent_answer,
            )
            if recovered_choice:
                predicted_choice = recovered_choice
                final_answer = raw_agent_answer
                final_answer_confidence = _extract_answer_confidence(final_answer)
                posthoc_choice_recovery_used = True
        return EvalResult(
            setting=self.setting_name,
            article_name=self.article_name,
            run_index=run_index,
            question_id=str(row.get("question_id", "") or ""),
            question=str(row.get("question", "") or ""),
            reference_choice=str(row.get("answer_choice", "") or "").strip().upper(),
            reference_answer=str(row.get("answer_text", "") or ""),
            predicted_choice=predicted_choice,
            predicted_answer=final_answer,
            is_correct=bool(predicted_choice and predicted_choice == str(row.get("answer_choice", "") or "").strip().upper()),
            latency_ms=int((time.time() - started) * 1000),
            extra={
                "tool_call_count": len(tool_uses),
                "tool_names": [
                    str((item or {}).get("tool_name", "") or "").strip()
                    for item in tool_uses
                    if str((item or {}).get("tool_name", "") or "").strip()
                ],
                "tool_uses": tool_uses,
                "strategy_context": strategy_context,
                "raw_agent_answer": raw_agent_answer,
                "open_answer_choice_adapter_enabled": self.open_answer_choice_adapter_enabled,
                "final_answer_confidence": final_answer_confidence,
                "low_confidence_fallback_used": low_confidence_fallback_used,
                "mcq_answer_adapter_used": mcq_answer_adapter_used,
                "terminal_mcq_enforcement_used": terminal_mcq_enforcement_used,
                "posthoc_choice_recovery_used": posthoc_choice_recovery_used,
                "error": error_text,
                "required_retrieval_enforced": True,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


def _evaluate_online_setting_incremental(
    *,
    base_cfg: KAGConfig,
    run_root: Path,
    report_root: Path,
    workspace_dir: Path,
    article_name: str,
    qa_rows: List[Dict[str, str]],
    setting_name: str,
    subagent_enabled: bool,
    online_repeat_states: Sequence[OnlineRepeatState],
    self_bootstrap_max_questions: int,
    warmup_questions: int,
    batch_size: int,
    online_attempts_per_question: int,
    eval_max_workers: int,
    enable_sql_tools: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    batches = _build_online_batches(
        qa_rows,
        warmup_questions=warmup_questions,
        batch_size=batch_size,
    )
    setting_results: List[Dict[str, Any]] = []
    training_reports: List[Dict[str, Any]] = []
    repeats = len(list(online_repeat_states))
    online_training_root_rel = (
        run_root.relative_to(REPO_ROOT)
        / "online_training_incremental"
        / "article_runs"
        / setting_name
        / article_name
    )
    progress_path = report_root / "progress" / setting_name / article_name / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    total_eval_attempts = len(qa_rows) * repeats

    for run_index, repeat_state in enumerate(online_repeat_states):
        for batch_index, batch_rows in enumerate(batches, start=1):
            _write_setting_progress(
                path=progress_path,
                setting_name=setting_name,
                article_name=article_name,
                repeats=repeats,
                repeat_index=run_index + 1,
                batch_index=batch_index,
                batch_total=len(batches),
                phase="evaluating_batch",
                evaluated_attempts_done=len(setting_results),
                evaluated_attempts_total=total_eval_attempts,
                batch_question_count=len(batch_rows),
                note="starting_batch_evaluation",
            )
            eval_cfg = _build_article_config(
                base_cfg,
                workspace_dir=workspace_dir,
                runtime_library_path=repeat_state.runtime_library_path,
                runtime_tool_metadata_dir=repeat_state.runtime_tool_metadata_dir,
                strategy_enabled=True,
                subagent_enabled=subagent_enabled,
                online_runtime_mode=True,
            )
            eval_cfg.strategy_memory.min_sampling_branches = max(
                5,
                int(getattr(eval_cfg.strategy_memory, "min_sampling_branches", 5) or 5),
            )
            eval_cfg.strategy_memory.self_bootstrap_sampling_attempts = max(
                3,
                int(getattr(eval_cfg.strategy_memory, "self_bootstrap_sampling_attempts", 3) or 3),
            )
            evaluator = AgentEvaluator(
                eval_cfg,
                setting_name=setting_name,
                article_name=article_name,
                enable_sql_tools=enable_sql_tools,
            )
            try:
                batch_results = _evaluate_rows_for_run(
                    rows=batch_rows,
                    evaluator=evaluator,
                    run_index=run_index,
                    max_workers=eval_max_workers,
                )
                setting_results.extend(batch_results)
            finally:
                evaluator.close()

            batch_tag = f"batch_{batch_index:03d}"
            online_training_csv = _write_online_training_csv(
                run_root
                / "data"
                / "online_training_csvs_incremental"
                / setting_name
                / article_name
                / f"repeat_{run_index}"
                / f"{batch_tag}.csv",
                batch_rows,
            )
            batch_dataset_name = f"{article_name}_{setting_name}_r{run_index}_{batch_tag}"
            batch_runtime_root = (
                run_root
                / "online_training_incremental"
                / "runtime_artifacts"
                / setting_name
                / article_name
                / f"repeat_{run_index}"
                / batch_tag
            )
            batch_runtime_library = batch_runtime_root / "strategy_library.json"
            batch_runtime_tool_meta = batch_runtime_root / "tool_metadata"
            online_train_cfg = _build_article_config(
                base_cfg,
                workspace_dir=workspace_dir,
                runtime_library_path=batch_runtime_library,
                runtime_tool_metadata_dir=batch_runtime_tool_meta,
                strategy_enabled=False,
                subagent_enabled=False,
            )
            online_train_cfg.strategy_memory.training_max_workers = max(
                1,
                min(
                    int(getattr(online_train_cfg.strategy_memory, "training_max_workers", 5) or 5),
                    len(batch_rows),
                ),
            )
            online_train_cfg.strategy_memory.min_sampling_branches = max(
                5,
                int(getattr(online_train_cfg.strategy_memory, "min_sampling_branches", 5) or 5),
            )
            online_train_cfg.strategy_memory.self_bootstrap_sampling_attempts = max(
                3,
                int(getattr(online_train_cfg.strategy_memory, "self_bootstrap_sampling_attempts", 3) or 3),
            )
            runner = OnlineStrategyTrainingRunner(
                config=online_train_cfg,
                csv_path=str(online_training_csv),
                dataset_name=batch_dataset_name,
                attempts_per_question=max(1, int(online_attempts_per_question or 1)),
                output_root=str(online_training_root_rel),
                runtime_library_path=str(batch_runtime_library),
                enable_sql_tools=enable_sql_tools,
                self_bootstrap_max_questions=self_bootstrap_max_questions,
            )
            try:
                result = runner.run()
            finally:
                runner.close()
            _write_setting_progress(
                path=progress_path,
                setting_name=setting_name,
                article_name=article_name,
                repeats=repeats,
                repeat_index=run_index + 1,
                batch_index=batch_index,
                batch_total=len(batches),
                phase="completed_batch_training",
                evaluated_attempts_done=len(setting_results),
                evaluated_attempts_total=total_eval_attempts,
                batch_question_count=len(batch_rows),
                training_progress_path=str(result.get("progress_path", "") or ""),
                note="completed_batch_training",
            )

            raw_templates = load_json(result["raw_template_path"])
            template_rows = raw_templates if isinstance(raw_templates, list) else []
            tool_candidates = load_json(result["tool_description_candidate_path"])
            if isinstance(tool_candidates, list):
                repeat_state.accumulator.add_tool_candidates(tool_candidates)
            if template_rows:
                repeat_state.accumulator.add_templates(
                    template_rows,
                    template_prefix=f"{setting_name}__{article_name}__r{run_index}__{batch_tag}",
                )
            runtime_summary = repeat_state.accumulator.export(
                consolidate=True,
                export_name=f"{article_name}__after_{batch_tag}",
            )
            training_reports.append(
                {
                    "article_name": article_name,
                    "setting": setting_name,
                    "run_index": run_index,
                    "batch_index": batch_index,
                    "batch_size": len(batch_rows),
                    "question_ids": [str(row.get("question_id", "") or "") for row in batch_rows],
                    "questions": [str(row.get("question", "") or "") for row in batch_rows],
                    "warmup_phase": False,
                    "runtime_summary": runtime_summary,
                    "active_runtime_library_path": str(repeat_state.runtime_library_path),
                    "active_runtime_tool_metadata_dir": str(repeat_state.runtime_tool_metadata_dir),
                    "training_csv": str(online_training_csv),
                    **result,
                }
            )
        checkpoint_dir = _checkpoint_online_repeat_state(repeat_state, article_name=article_name)
        training_reports.append(
            {
                "article_name": article_name,
                "setting": setting_name,
                "run_index": run_index,
                "batch_index": len(batches),
                "batch_size": 0,
                "question_ids": [],
                "questions": [],
                "warmup_phase": False,
                "runtime_checkpoint_dir": str(checkpoint_dir),
                "checkpoint_only": True,
            }
        )

    setting_results.sort(key=lambda item: (item["question_id"], int(item["run_index"])))
    _write_setting_progress(
        path=progress_path,
        setting_name=setting_name,
        article_name=article_name,
        repeats=repeats,
        repeat_index=repeats,
        batch_index=len(batches),
        batch_total=len(batches),
        phase="completed",
        evaluated_attempts_done=len(setting_results),
        evaluated_attempts_total=total_eval_attempts,
        batch_question_count=0,
        note="completed_setting",
    )
    return setting_results, training_reports


class TraditionalHybridThreadLocal:
    def __init__(self, cfg: KAGConfig) -> None:
        self.cfg = cfg
        self.local = threading.local()

    def _build_state(self) -> Dict[str, Any]:
        prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        llm = OpenAILLM(self.cfg, llm_profile="retriever")
        doc_vs = VectorStore(self.cfg, "document")
        sent_vs = VectorStore(self.cfg, "sentence")
        base = Path(self.cfg.knowledge_graph_builder.file_path) / "all_document_chunks.json"
        data = json.loads(base.read_text(encoding="utf-8"))
        docs: List[LCDocument] = []
        keys_to_drop = {"chunk_index", "chunk_type", "doc_title", "order", "total_doc_chunks"}
        for item in data:
            chunk_id = item.get("id")
            content = str(item.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            meta = dict(item.get("metadata") or {})
            meta["chunk_id"] = chunk_id
            for key in list(keys_to_drop):
                meta.pop(key, None)
            docs.append(LCDocument(page_content=content, metadata=meta))
        split_docs = _split_bm25_documents(
            docs,
            chunk_size=int(getattr(self.cfg.document_processing, "bm25_chunk_size", 250) or 0),
        )
        bm25 = KeywordBM25Retriever(split_docs, k_default=10)
        return {
            "prompt_loader": prompt_loader,
            "llm": llm,
            "doc_vs": doc_vs,
            "sent_vs": sent_vs,
            "bm25": bm25,
        }

    def state(self) -> Dict[str, Any]:
        state = getattr(self.local, "state", None)
        if state is None:
            state = self._build_state()
            self.local.state = state
        return state


class TraditionalHybridEvaluator:
    def __init__(self, cfg: KAGConfig, *, article_name: str) -> None:
        self.cfg = cfg
        self.article_name = article_name
        self.tlocal = TraditionalHybridThreadLocal(cfg)
        self.choice_extractor = ChoiceExtractor(cfg)

    def evaluate_row(self, row: Dict[str, str], run_index: int) -> EvalResult:
        started = time.time()
        state = self.tlocal.state()
        error_text = ""
        final_answer = ""
        fused: List[Dict[str, Any]] = []
        predicted_choice = ""
        try:
            doc_hits = state["doc_vs"].search(_format_mcq_for_agent(row["question"]), limit=8)
            sent_hits = state["sent_vs"].search(_format_mcq_for_agent(row["question"]), limit=8)
            bm25_hits = state["bm25"].retrieve(_format_mcq_for_agent(row["question"]), k=8)
            fused = _rrf_merge(doc_hits, sent_hits, bm25_hits, top_k=8)
            evidence_text = "\n\n".join(_format_evidence_item(i, item) for i, item in enumerate(fused, start=1))
            prompt = state["prompt_loader"].render(
                "memory/answer_with_retrieved_evidence",
                task_values={
                    "question": _format_mcq_for_agent(row["question"]),
                    "retrieved_evidence": evidence_text or "(none)",
                },
                strict=True,
            )
            llm_result = state["llm"].run([{"role": "user", "content": prompt}])
            final_answer = _extract_llm_text(llm_result)
            predicted_choice = self.choice_extractor.choose(question_text=row["question"], answer_text=final_answer)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
        return EvalResult(
            setting="traditional_hybrid_rag_bm25",
            article_name=self.article_name,
            run_index=run_index,
            question_id=str(row.get("question_id", "") or ""),
            question=str(row.get("question", "") or ""),
            reference_choice=str(row.get("answer_choice", "") or "").strip().upper(),
            reference_answer=str(row.get("answer_text", "") or ""),
            predicted_choice=predicted_choice,
            predicted_answer=final_answer,
            is_correct=bool(predicted_choice and predicted_choice == str(row.get("answer_choice", "") or "").strip().upper()),
            latency_ms=int((time.time() - started) * 1000),
            extra={
                "evidence": fused,
                "error": error_text,
            },
        )


def _evaluate_setting(
    *,
    rows: List[Dict[str, str]],
    evaluator: Any,
    repeats: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    tasks: List[Tuple[Dict[str, str], int]] = []
    for row in rows:
        for run_index in range(repeats):
            tasks.append((row, run_index))
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(evaluator.evaluate_row, row, run_index): (row, run_index) for row, run_index in tasks}
        for future in as_completed(future_map):
            results.append(future.result().to_dict())
    results.sort(key=lambda item: (item["question_id"], int(item["run_index"])))
    return results


def _evaluate_rows_for_run(
    *,
    rows: List[Dict[str, str]],
    evaluator: Any,
    run_index: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    results: List[Dict[str, Any]] = []
    worker_count = max(1, min(int(max_workers or 1), len(rows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(evaluator.evaluate_row, row, run_index): row for row in rows}
        for future in as_completed(future_map):
            results.append(future.result().to_dict())
    results.sort(key=lambda item: (item["question_id"], int(item["run_index"])))
    return results


def _write_setting_progress(
    *,
    path: Path,
    setting_name: str,
    article_name: str,
    repeats: int,
    repeat_index: int,
    batch_index: int,
    batch_total: int,
    phase: str,
    evaluated_attempts_done: int,
    evaluated_attempts_total: int,
    batch_question_count: int,
    training_progress_path: str = "",
    note: str = "",
) -> None:
    payload = {
        "setting": setting_name,
        "article_name": article_name,
        "repeats": int(repeats or 0),
        "repeat_index": int(repeat_index or 0),
        "batch_index": int(batch_index or 0),
        "batch_total": int(batch_total or 0),
        "phase": str(phase or "").strip(),
        "evaluated_attempts_done": int(evaluated_attempts_done or 0),
        "evaluated_attempts_total": int(evaluated_attempts_total or 0),
        "batch_question_count": int(batch_question_count or 0),
        "training_progress_path": str(training_progress_path or "").strip(),
        "note": str(note or "").strip(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if evaluated_attempts_total > 0:
        payload["overall_progress"] = round(float(evaluated_attempts_done) / float(evaluated_attempts_total), 4)
    json_dump_atomic(str(path), payload)


def _summarize_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    total_attempts = len(results)
    correct_attempts = sum(1 for item in results if bool(item.get("is_correct", False)))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        article_name = str(item.get("article_name", "") or "").strip()
        question_id = str(item.get("question_id", "") or "").strip()
        key = f"{article_name}::{question_id}" if article_name else question_id
        grouped.setdefault(key, []).append(item)
    pass_questions = sum(1 for attempts in grouped.values() if any(bool(item.get("is_correct", False)) for item in attempts))
    avg_latency_ms = int(sum(int(item.get("latency_ms", 0) or 0) for item in results) / float(total_attempts or 1))
    return {
        "question_count": question_count,
        "repeats": repeats,
        "total_attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "overall_accuracy": round(correct_attempts / float(total_attempts or 1), 4),
        "pass_accuracy": round(pass_questions / float(question_count or 1), 4),
        "pass_question_count": pass_questions,
        "avg_latency_ms": avg_latency_ms,
    }


def _format_setting_repeats_text(setting_repeats: Dict[str, Any], setting_order: Sequence[str]) -> str:
    pairs: List[str] = []
    for setting in setting_order:
        if setting not in setting_repeats:
            continue
        try:
            repeat_value = int(setting_repeats.get(setting, 0) or 0)
        except Exception:
            repeat_value = 0
        if repeat_value <= 0:
            continue
        pairs.append(f"{setting}={repeat_value}")
    return ", ".join(pairs)


def _summarize_articles(
    *,
    setting_results: Dict[str, List[Dict[str, Any]]],
    repeats: int,
) -> Dict[str, Any]:
    article_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for setting, rows in setting_results.items():
        for row in rows:
            article_map.setdefault(str(row.get("article_name", "") or ""), {}).setdefault(setting, []).append(row)
    payload: Dict[str, Any] = {}
    for article_name, by_setting in sorted(article_map.items()):
        article_summary: Dict[str, Any] = {}
        question_count = len(
            {
                f"{str(item.get('article_name', '') or '').strip()}::{str(item.get('question_id', '') or '').strip()}"
                for items in by_setting.values()
                for item in items
                if str(item.get("question_id", "") or "").strip()
            }
        )
        for setting, rows in by_setting.items():
            article_summary[setting] = _summarize_setting(rows, question_count=question_count, repeats=repeats)
        payload[article_name] = article_summary
    return payload


def _copy_runtime_backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_runtime_tree(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def _materialize_runtime_snapshot(*, source_dir: Path, runtime_library_path: Path, runtime_tool_metadata_dir: Path) -> None:
    source_library = source_dir / "strategy_library.json"
    if not source_library.exists():
        raise FileNotFoundError(f"Missing strategy_library.json under {source_dir}")
    runtime_library_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_library, runtime_library_path)
    source_question_hints = source_dir / "source_question_hints.json"
    target_source_question_hints = runtime_library_path.parent / "source_question_hints.json"
    if source_question_hints.exists():
        shutil.copy2(source_question_hints, target_source_question_hints)
    else:
        json_dump_atomic(str(target_source_question_hints), {})
    source_template_index = source_dir / "template_source_index.json"
    target_template_index = runtime_library_path.parent / "template_source_index.json"
    if source_template_index.exists():
        shutil.copy2(source_template_index, target_template_index)
    else:
        json_dump_atomic(str(target_template_index), {})
    source_tool_meta = source_dir / "tool_metadata"
    if source_tool_meta.exists():
        _copy_runtime_tree(source_tool_meta, runtime_tool_metadata_dir)
    else:
        runtime_manager = StrategyRuntimeAssetManager(
            library_path=str(runtime_library_path),
            tool_metadata_runtime_dir=str(runtime_tool_metadata_dir),
        )
        runtime_manager.clear_tool_metadata()


def _write_runtime_snapshot(
    *,
    runtime_library_path: Path,
    runtime_tool_metadata_dir: Path,
    snapshot_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not runtime_library_path.exists():
        raise FileNotFoundError(f"Missing runtime library: {runtime_library_path}")
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime_library_path, snapshot_dir / "strategy_library.json")
    source_question_hints = runtime_library_path.parent / "source_question_hints.json"
    if source_question_hints.exists():
        shutil.copy2(source_question_hints, snapshot_dir / "source_question_hints.json")
    else:
        json_dump_atomic(str(snapshot_dir / "source_question_hints.json"), {})
    source_template_index = runtime_library_path.parent / "template_source_index.json"
    if source_template_index.exists():
        shutil.copy2(source_template_index, snapshot_dir / "template_source_index.json")
    else:
        json_dump_atomic(str(snapshot_dir / "template_source_index.json"), {})
    target_tool_meta = snapshot_dir / "tool_metadata"
    if runtime_tool_metadata_dir.exists():
        _copy_runtime_tree(runtime_tool_metadata_dir, target_tool_meta)
    else:
        runtime_manager = StrategyRuntimeAssetManager(
            library_path=str(snapshot_dir / "strategy_library.json"),
            tool_metadata_runtime_dir=str(target_tool_meta),
        )
        runtime_manager.clear_tool_metadata()
    json_dump_atomic(
        str(snapshot_dir / "checkpoint_meta.json"),
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {}),
        },
    )


def _materialize_external_offline_runtime(*, source_dir: Path, runtime_library_path: Path, runtime_tool_metadata_dir: Path) -> None:
    _materialize_runtime_snapshot(
        source_dir=source_dir,
        runtime_library_path=runtime_library_path,
        runtime_tool_metadata_dir=runtime_tool_metadata_dir,
    )


def _build_online_repeat_states(
    *,
    base_cfg: KAGConfig,
    runtime_root: Path,
    report_root: Path,
    run_name: str,
    selected_settings: Sequence[str],
    setting_repeats: Dict[str, int],
) -> Dict[str, List[OnlineRepeatState]]:
    states: Dict[str, List[OnlineRepeatState]] = {}
    for setting_name in selected_settings:
        if setting_name not in {"online_strategy_agent", "online_strategy_subagent"}:
            continue
        repeat_states: List[OnlineRepeatState] = []
        for repeat_index in range(max(0, int(setting_repeats.get(setting_name, 0) or 0))):
            repeat_root = runtime_root / "online_incremental" / setting_name / f"repeat_{repeat_index}"
            runtime_library_path = repeat_root / "strategy_library.json"
            runtime_tool_metadata_dir = repeat_root / "tool_metadata"
            checkpoint_root = runtime_root / "online_checkpoints" / setting_name / f"repeat_{repeat_index}"
            accumulator = StrategyLibraryAccumulator(
                cfg=base_cfg,
                runtime_library_path=runtime_library_path,
                runtime_tool_metadata_dir=runtime_tool_metadata_dir,
                output_dir=report_root / "online_library_build" / setting_name / f"repeat_{repeat_index}",
                dataset_name=f"{run_name}_{setting_name}_repeat_{repeat_index}",
            )
            repeat_states.append(
                OnlineRepeatState(
                    setting_name=setting_name,
                    repeat_index=repeat_index,
                    runtime_root=repeat_root,
                    runtime_library_path=runtime_library_path,
                    runtime_tool_metadata_dir=runtime_tool_metadata_dir,
                    checkpoint_root=checkpoint_root,
                    accumulator=accumulator,
                )
            )
        states[setting_name] = repeat_states
    return states


def _reset_online_repeat_states(states: Dict[str, List[OnlineRepeatState]], *, clear_checkpoints: bool) -> None:
    for repeat_states in states.values():
        for state in repeat_states:
            state.accumulator.reset_state(clear_runtime=True)
            state.last_checkpoint_article = ""
            if clear_checkpoints and state.checkpoint_root.exists():
                shutil.rmtree(state.checkpoint_root)


def _restore_online_repeat_state_from_checkpoint(state: OnlineRepeatState, *, article_name: str) -> Path:
    checkpoint_dir = state.checkpoint_dir(article_name)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Missing online checkpoint for setting={state.setting_name} repeat={state.repeat_index} article={article_name}: {checkpoint_dir}"
        )
    _materialize_runtime_snapshot(
        source_dir=checkpoint_dir,
        runtime_library_path=state.runtime_library_path,
        runtime_tool_metadata_dir=state.runtime_tool_metadata_dir,
    )
    state.accumulator.hydrate_from_runtime()
    state.last_checkpoint_article = article_name
    return checkpoint_dir


def _checkpoint_online_repeat_state(state: OnlineRepeatState, *, article_name: str) -> Path:
    checkpoint_dir = state.checkpoint_dir(article_name)
    _write_runtime_snapshot(
        runtime_library_path=state.runtime_library_path,
        runtime_tool_metadata_dir=state.runtime_tool_metadata_dir,
        snapshot_dir=checkpoint_dir,
        metadata={
            "setting_name": state.setting_name,
            "repeat_index": state.repeat_index,
            "article_name": article_name,
        },
    )
    state.last_checkpoint_article = article_name
    return checkpoint_dir


def _summarize_online_repeat_states(states: Dict[str, List[OnlineRepeatState]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for setting_name, repeat_states in states.items():
        per_repeat: Dict[str, Any] = {}
        template_counts: List[int] = []
        tool_override_counts: List[int] = []
        for state in repeat_states:
            summary = _load_existing_runtime_summary(
                runtime_library_path=state.runtime_library_path,
                runtime_tool_metadata_dir=state.runtime_tool_metadata_dir,
                reused=bool(state.last_checkpoint_article),
            )
            summary["repeat_index"] = state.repeat_index
            summary["last_checkpoint_article"] = state.last_checkpoint_article
            per_repeat[f"repeat_{state.repeat_index}"] = summary
            template_counts.append(int(summary.get("template_count", 0) or 0))
            tool_override_counts.append(int(summary.get("tool_override_count", 0) or 0))
        repeat_count = len(repeat_states)
        summaries[setting_name] = {
            "repeat_count": repeat_count,
            "template_count_total": sum(template_counts),
            "template_count_avg": round(sum(template_counts) / repeat_count, 4) if repeat_count else 0.0,
            "tool_override_count_total": sum(tool_override_counts),
            "tool_override_count_avg": round(sum(tool_override_counts) / repeat_count, 4) if repeat_count else 0.0,
            "per_repeat": per_repeat,
        }
    return summaries


def _load_existing_runtime_summary(
    *,
    runtime_library_path: Path,
    runtime_tool_metadata_dir: Path,
    reused: bool,
) -> Dict[str, Any]:
    template_count = 0
    dataset_name = ""
    generated_at = ""
    if runtime_library_path.exists():
        try:
            payload = json.loads(runtime_library_path.read_text(encoding="utf-8"))
            template_count = int(payload.get("template_count", 0) or 0)
            dataset_name = str(payload.get("dataset_name", "") or "")
            generated_at = str(payload.get("generated_at", "") or "")
        except Exception:
            pass
    tool_override_count = 0
    if runtime_tool_metadata_dir.exists():
        tool_override_count = len(list(runtime_tool_metadata_dir.rglob("*.json")))
    return {
        "reused_existing_runtime": bool(reused),
        "runtime_library_path": str(runtime_library_path),
        "runtime_tool_metadata_dir": str(runtime_tool_metadata_dir),
        "template_count": template_count,
        "tool_override_count": tool_override_count,
        "dataset_name": dataset_name,
        "generated_at": generated_at,
    }


def _setting_payload_is_complete(payload: Optional[Dict[str, Any]], *, expected_question_count: int, repeats: int) -> bool:
    if not isinstance(payload, dict):
        return False
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    expected_attempts = int(expected_question_count or 0) * int(repeats or 0)
    if len(results) != expected_attempts:
        return False
    question_count = int(payload.get("question_count", 0) or 0)
    payload_repeats = int(payload.get("repeats", 0) or 0)
    return question_count == int(expected_question_count or 0) and payload_repeats == int(repeats or 0)


def _default_workspace_asset_root() -> Path:
    return REPO_ROOT / "experiments" / "quality" / "assets" / "article_workspaces"


def _resolve_article_workspace_dir(
    *,
    article_name: str,
    workspace_root: Path,
    workspace_source_root: Optional[Path],
    workspace_asset_root: Optional[Path],
) -> Path:
    if workspace_source_root is not None:
        return workspace_source_root / article_name
    if workspace_asset_root is not None:
        return workspace_asset_root / article_name
    return workspace_root / article_name


def _update_workspace_asset_registry(
    *,
    workspace_asset_root: Optional[Path],
    article_name: str,
    workspace_dir: Path,
    article_json_path: Optional[Path] = None,
) -> None:
    asset_root = workspace_asset_root.resolve() if workspace_asset_root is not None else None
    target_dir = workspace_dir.resolve()
    if asset_root is None or target_dir.parent != asset_root:
        return
    registry_path = asset_root / "_workspace_registry.json"
    lock_path = asset_root / "._workspace_registry.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload: Dict[str, Any] = {}
            if registry_path.exists():
                try:
                    loaded = json.loads(registry_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = {}
            articles = payload.get("articles") if isinstance(payload.get("articles"), dict) else {}
            marker_path = workspace_dir / "build_marker.json"
            marker_payload: Dict[str, Any] = {}
            if marker_path.exists():
                try:
                    marker_loaded = json.loads(marker_path.read_text(encoding="utf-8"))
                    if isinstance(marker_loaded, dict):
                        marker_payload = marker_loaded
                except Exception:
                    marker_payload = {}
            articles[str(article_name or "").strip()] = {
                "workspace_dir": str(workspace_dir),
                "article_json_path": str(article_json_path) if article_json_path is not None else str(marker_payload.get("json_file_path", "") or ""),
                "built_at": str(marker_payload.get("built_at", "") or time.strftime("%Y-%m-%d %H:%M:%S")),
                "marker_path": str(marker_path),
            }
            payload["articles"] = articles
            payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json_dump_atomic(str(registry_path), payload)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _prepare_article_cfg(
    *,
    base_cfg: KAGConfig,
    article_name: str,
    article_json_path: Path,
    workspace_dir: Path,
    workspace_asset_root: Optional[Path] = None,
    runtime_library_path: Optional[Path] = None,
    runtime_tool_metadata_dir: Optional[Path] = None,
    strategy_enabled: bool = False,
    subagent_enabled: bool = False,
    rebuild: bool = False,
    reuse_existing_workspace_only: bool = False,
) -> KAGConfig:
    if reuse_existing_workspace_only:
        cfg = load_existing_article_workspace_or_raise(
            base_cfg=base_cfg,
            workspace_dir=workspace_dir,
            runtime_library_path=runtime_library_path,
            runtime_tool_metadata_dir=runtime_tool_metadata_dir,
            strategy_enabled=strategy_enabled,
            subagent_enabled=subagent_enabled,
        )
        _ensure_workspace_vector_stores_current(cfg)
        _update_workspace_asset_registry(
            workspace_asset_root=workspace_asset_root,
            article_name=article_name,
            workspace_dir=workspace_dir,
            article_json_path=article_json_path,
        )
        return cfg
    cfg = ensure_article_ready(
        base_cfg=base_cfg,
        article_json_path=article_json_path,
        workspace_dir=workspace_dir,
        runtime_library_path=runtime_library_path,
        runtime_tool_metadata_dir=runtime_tool_metadata_dir,
        strategy_enabled=strategy_enabled,
        subagent_enabled=subagent_enabled,
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


QUALITY_SETTING_ORDER = [
    "no_strategy_agent",
    "traditional_hybrid_rag_bm25",
    "offline_strategy_agent",
    "offline_strategy_subagent",
    "online_strategy_agent",
    "online_strategy_subagent",
]

QUALITY_SETTING_DEFAULT_REPEATS = {
    "no_strategy_agent": 5,
    "traditional_hybrid_rag_bm25": 5,
    "offline_strategy_agent": 1,
    "offline_strategy_subagent": 1,
    "online_strategy_agent": 1,
    "online_strategy_subagent": 1,
}

EXPERIMENT_SETTING_GROUPS = {
    "all": list(QUALITY_SETTING_ORDER),
    "exp1": [
        "no_strategy_agent",
        "traditional_hybrid_rag_bm25",
    ],
    "exp2": [
        "offline_strategy_agent",
        "offline_strategy_subagent",
        "online_strategy_agent",
        "online_strategy_subagent",
    ],
}


def _resolve_cli_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _resolve_selected_settings(*, experiment: str, settings_arg: str) -> List[str]:
    if str(settings_arg or "").strip():
        requested = [item.strip() for item in str(settings_arg or "").split(",") if item.strip()]
    else:
        requested = list(EXPERIMENT_SETTING_GROUPS.get(str(experiment or "all").strip().lower(), []))
    if not requested:
        raise ValueError("No settings selected. Provide --experiment or --settings.")
    unknown = [name for name in requested if name not in QUALITY_SETTING_ORDER]
    if unknown:
        raise ValueError(f"Unknown settings requested: {unknown}")
    deduped: List[str] = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return deduped


def _resolve_setting_repeats(
    *,
    selected_settings: Sequence[str],
    global_repeats: int,
    setting_repeats_arg: str,
) -> Dict[str, int]:
    setting_repeats: Dict[str, int] = {
        str(name): int(QUALITY_SETTING_DEFAULT_REPEATS.get(str(name), 1) or 1)
        for name in selected_settings
    }
    if int(global_repeats or 0) > 0:
        override = int(global_repeats)
        setting_repeats = {str(name): override for name in selected_settings}
    raw_arg = str(setting_repeats_arg or "").strip()
    if raw_arg:
        for chunk in raw_arg.split(","):
            item = str(chunk or "").strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid --setting-repeats item: {item}")
            name, value = item.split("=", 1)
            setting_name = str(name or "").strip()
            if setting_name not in setting_repeats:
                raise ValueError(f"Unknown setting in --setting-repeats: {setting_name}")
            repeat_value = int(str(value or "").strip() or "0")
            if repeat_value <= 0:
                raise ValueError(f"Repeat count must be positive for {setting_name}")
            setting_repeats[setting_name] = repeat_value
    return setting_repeats


def _empty_setting_results(setting_order: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {name: [] for name in setting_order}


def _merge_setting_order(existing_order: Sequence[str], current_order: Sequence[str], result_map: Dict[str, Any]) -> List[str]:
    available = {str(name) for name in result_map.keys()}
    merged: List[str] = []
    for name in list(existing_order or []) + list(current_order or []) + list(QUALITY_SETTING_ORDER):
        key = str(name)
        if key not in available or key in merged:
            continue
        merged.append(key)
    return merged


def _load_existing_article_payload(report_root: Path, article_name: str) -> Optional[Dict[str, Any]]:
    path = report_root / "article_results" / article_name / "result.json"
    if not path.exists():
        return None
    try:
        payload = load_json(str(path))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_existing_setting_payload(report_root: Path, article_name: str, setting_name: str) -> Optional[Dict[str, Any]]:
    path = report_root / "article_results" / article_name / "settings" / f"{setting_name}.json"
    if not path.exists():
        return None
    try:
        payload = load_json(str(path))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_article_result_payload(
    *,
    article_name: str,
    setting_results: Dict[str, List[Dict[str, Any]]],
    online_training_reports: List[Dict[str, Any]],
    setting_repeats: Dict[str, int],
    setting_order: Sequence[str],
) -> Dict[str, Any]:
    question_count = len(
        {
            f"{str(item.get('article_name', article_name) or '').strip()}::{str(item.get('question_id', '') or '').strip()}"
            for rows in setting_results.values()
            for item in (rows or [])
            if str(item.get("question_id", "") or "").strip()
        }
    )
    summary = {
        setting: _summarize_setting(rows, question_count=question_count, repeats=int(setting_repeats.get(setting, 1) or 1))
        for setting, rows in setting_results.items()
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "article_name": article_name,
        "repeats": max([int(setting_repeats.get(setting, 0) or 0) for setting in setting_results.keys()] or [0]),
        "question_count": question_count,
        "setting_repeats": {str(setting): int(setting_repeats.get(setting, 1) or 1) for setting in setting_results.keys()},
        "setting_order": list(setting_order),
        "summary": summary,
        "results": setting_results,
        "online_training_reports": online_training_reports,
    }


def _write_article_result_files(
    *,
    report_root: Path,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    article_name = str(payload.get("article_name", "") or "").strip()
    article_dir = report_root / "article_results" / article_name
    article_dir.mkdir(parents=True, exist_ok=True)
    json_path = article_dir / "result.json"
    md_path = article_dir / "report.md"
    json_dump_atomic(str(json_path), payload)

    lines: List[str] = [
        f"# QUALITY Article Report: {article_name}",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Article: `{article_name}`",
        f"- Question count: `{payload.get('question_count', 0)}`",
        "",
        "## Summary",
        "",
    ]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    setting_order = list(payload.get("setting_order") or QUALITY_SETTING_ORDER)
    setting_repeats = payload.get("setting_repeats") if isinstance(payload.get("setting_repeats"), dict) else {}
    repeats_text = _format_setting_repeats_text(setting_repeats, setting_order)
    if repeats_text:
        lines.insert(5, f"- Setting repeats: `{repeats_text}`")
    else:
        lines.insert(5, f"- Repeats: `{payload.get('repeats', 0)}`")
    for setting in setting_order:
        row = summary.get(setting) if isinstance(summary, dict) else None
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"### {setting}",
                "",
                f"- Total attempts: `{row.get('total_attempts', 0)}`",
                f"- Correct attempts: `{row.get('correct_attempts', 0)}`",
                f"- Overall accuracy: `{row.get('overall_accuracy', 0)}`",
                f"- Pass accuracy: `{row.get('pass_accuracy', 0)}`",
                f"- Pass question count: `{row.get('pass_question_count', 0)}`",
                f"- Average latency: `{row.get('avg_latency_ms', 0)} ms`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path)}


def _build_setting_result_payload(
    *,
    article_name: str,
    setting_name: str,
    results: List[Dict[str, Any]],
    repeats: int,
    online_training_reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    question_count = len({str(item.get("question_id", "") or "") for item in results if str(item.get("question_id", "") or "")})
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "article_name": article_name,
        "setting_name": setting_name,
        "repeats": repeats,
        "question_count": question_count,
        "summary": _summarize_setting(results, question_count=question_count, repeats=repeats),
        "results": results,
        "online_training_reports": list(online_training_reports or []),
    }


def _write_setting_result_files(
    *,
    report_root: Path,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    article_name = str(payload.get("article_name", "") or "").strip()
    setting_name = str(payload.get("setting_name", "") or "").strip()
    article_dir = report_root / "article_results" / article_name / "settings"
    article_dir.mkdir(parents=True, exist_ok=True)
    json_path = article_dir / f"{setting_name}.json"
    md_path = article_dir / f"{setting_name}.md"
    json_dump_atomic(str(json_path), payload)

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines: List[str] = [
        f"# QUALITY Setting Report: {article_name} / {setting_name}",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Article: `{article_name}`",
        f"- Setting: `{setting_name}`",
        f"- Question count: `{payload.get('question_count', 0)}`",
        f"- Repeats: `{payload.get('repeats', 0)}`",
        "",
        "## Summary",
        "",
        f"- Total attempts: `{summary.get('total_attempts', 0)}`",
        f"- Correct attempts: `{summary.get('correct_attempts', 0)}`",
        f"- Overall accuracy: `{summary.get('overall_accuracy', 0)}`",
        f"- Pass accuracy: `{summary.get('pass_accuracy', 0)}`",
        f"- Pass question count: `{summary.get('pass_question_count', 0)}`",
        f"- Average latency: `{summary.get('avg_latency_ms', 0)} ms`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path)}


def _flush_article_progress(
    *,
    report_root: Path,
    article_name: str,
    setting_results: Dict[str, List[Dict[str, Any]]],
    online_training_reports: List[Dict[str, Any]],
    setting_repeats: Dict[str, int],
    setting_order: Sequence[str],
) -> Dict[str, str]:
    populated_results = {name: rows for name, rows in setting_results.items() if rows}
    existing_payload = _load_existing_article_payload(report_root, article_name)
    merged_results: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(existing_payload, dict):
        existing_results = existing_payload.get("results") if isinstance(existing_payload.get("results"), dict) else {}
        for name, rows in existing_results.items():
            if isinstance(rows, list):
                merged_results[str(name)] = rows
    merged_setting_repeats: Dict[str, int] = {}
    if isinstance(existing_payload, dict):
        existing_setting_repeats = existing_payload.get("setting_repeats")
        if isinstance(existing_setting_repeats, dict):
            for name, value in existing_setting_repeats.items():
                try:
                    merged_setting_repeats[str(name)] = int(value or 0)
                except Exception:
                    continue
        existing_summary = existing_payload.get("summary") if isinstance(existing_payload.get("summary"), dict) else {}
        for name, row in existing_summary.items():
            if str(name) in merged_setting_repeats:
                continue
            if isinstance(row, dict):
                try:
                    merged_setting_repeats[str(name)] = int(row.get("repeats", 0) or 0)
                except Exception:
                    continue
    for name, value in setting_repeats.items():
        merged_setting_repeats[str(name)] = int(value or 0)
    merged_results.update(populated_results)
    current_settings = set(populated_results.keys())
    merged_online_reports: List[Dict[str, Any]] = []
    if isinstance(existing_payload, dict):
        existing_reports = existing_payload.get("online_training_reports")
        if isinstance(existing_reports, list):
            merged_online_reports.extend(
                [item for item in existing_reports if isinstance(item, dict) and str(item.get("setting", "") or "") not in current_settings]
            )
    merged_online_reports.extend(list(online_training_reports or []))
    merged_setting_order = _merge_setting_order(
        existing_payload.get("setting_order") if isinstance(existing_payload, dict) else [],
        setting_order,
        merged_results,
    )
    payload = _build_article_result_payload(
        article_name=article_name,
        setting_results=merged_results,
        online_training_reports=merged_online_reports,
        setting_repeats=merged_setting_repeats,
        setting_order=merged_setting_order,
    )
    return _write_article_result_files(report_root=report_root, payload=payload)


def _persist_setting_progress(
    *,
    report_root: Path,
    article_name: str,
    setting_name: str,
    results: List[Dict[str, Any]],
    article_setting_results: Dict[str, List[Dict[str, Any]]],
    online_training_reports: List[Dict[str, Any]],
    setting_repeats: Dict[str, int],
    setting_order: Sequence[str],
) -> None:
    setting_payload = _build_setting_result_payload(
        article_name=article_name,
        setting_name=setting_name,
        results=results,
        repeats=int(setting_repeats.get(setting_name, 1) or 1),
        online_training_reports=[item for item in online_training_reports if str(item.get("setting", "") or "") == setting_name],
    )
    _write_setting_result_files(report_root=report_root, payload=setting_payload)
    _flush_article_progress(
        report_root=report_root,
        article_name=article_name,
        setting_results=article_setting_results,
        online_training_reports=online_training_reports,
        setting_repeats=setting_repeats,
        setting_order=setting_order,
    )


def _load_article_result_payloads(report_root: Path) -> List[Dict[str, Any]]:
    article_root = report_root / "article_results"
    if not article_root.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    for path in sorted(article_root.glob("*/result.json")):
        try:
            payload = load_json(str(path))
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _resolve_report_setting_order(
    *,
    article_payloads: Sequence[Dict[str, Any]],
    current_setting_order: Sequence[str],
) -> List[str]:
    available: Dict[str, bool] = {str(name): True for name in current_setting_order}
    existing_order: List[str] = []
    for payload in article_payloads:
        if not isinstance(payload, dict):
            continue
        payload_order = payload.get("setting_order") if isinstance(payload.get("setting_order"), list) else []
        for name in payload_order:
            key = str(name)
            if key and key not in existing_order:
                existing_order.append(key)
        result_map = payload.get("results") if isinstance(payload.get("results"), dict) else {}
        for name, rows in result_map.items():
            if isinstance(rows, list) and rows:
                available[str(name)] = True
    return _merge_setting_order(existing_order, current_setting_order, available)


def _resolve_report_setting_repeats(
    *,
    article_payloads: Sequence[Dict[str, Any]],
    current_setting_repeats: Dict[str, int],
    setting_order: Sequence[str],
) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for payload in article_payloads:
        if not isinstance(payload, dict):
            continue
        payload_setting_repeats = payload.get("setting_repeats")
        if isinstance(payload_setting_repeats, dict):
            for name, value in payload_setting_repeats.items():
                try:
                    merged[str(name)] = int(value or 0)
                except Exception:
                    continue
        payload_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        for name, row in payload_summary.items():
            if str(name) in merged:
                continue
            if isinstance(row, dict):
                try:
                    merged[str(name)] = int(row.get("repeats", 0) or 0)
                except Exception:
                    continue
    for name, value in current_setting_repeats.items():
        merged[str(name)] = int(value or 0)
    return {name: int(merged.get(name, QUALITY_SETTING_DEFAULT_REPEATS.get(name, 1)) or 1) for name in setting_order}


def _aggregate_article_payloads(
    *,
    article_payloads: List[Dict[str, Any]],
    setting_repeats: Dict[str, int],
    setting_order: Sequence[str],
) -> Dict[str, Any]:
    setting_results = _empty_setting_results(setting_order)
    article_summary: Dict[str, Any] = {}
    online_training_reports: List[Dict[str, Any]] = []
    for payload in article_payloads:
        article_name = str(payload.get("article_name", "") or "").strip()
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        article_summary[article_name] = summary
        result_map = payload.get("results") if isinstance(payload.get("results"), dict) else {}
        for setting in setting_order:
            rows = result_map.get(setting) if isinstance(result_map, dict) else None
            if isinstance(rows, list):
                setting_results[setting].extend(rows)
        reports = payload.get("online_training_reports")
        if isinstance(reports, list):
            online_training_reports.extend([row for row in reports if isinstance(row, dict)])

    summary = {
        setting: _summarize_setting(
            setting_results.get(setting, []),
            question_count=len(
                {
                    f"{str(item.get('article_name', '') or '').strip()}::{str(item.get('question_id', '') or '').strip()}"
                    for item in setting_results.get(setting, [])
                    if str(item.get("question_id", "") or "").strip()
                }
            ),
            repeats=int(setting_repeats.get(setting, 1) or 1),
        )
        for setting in setting_order
    }
    return {
        "setting_order": list(setting_order),
        "setting_repeats": dict(setting_repeats),
        "summary": summary,
        "article_summary": article_summary,
        "setting_results": setting_results,
        "online_training_reports": online_training_reports,
    }


def _append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record_run_lifecycle_event(
    *,
    lifecycle_path: Path,
    status_path: Path,
    event: str,
    run_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "event": str(event or "").strip(),
        "run_name": str(run_name or "").strip(),
        "pid": os.getpid(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    _append_jsonl_row(lifecycle_path, payload)
    json_dump_atomic(str(status_path), payload)


def _install_run_lifecycle_logging(*, run_root: Path, run_name: str) -> Dict[str, Any]:
    lifecycle_root = run_root / "runtime_events"
    lifecycle_path = lifecycle_root / "lifecycle.jsonl"
    status_path = lifecycle_root / "last_status.json"
    lifecycle_root.mkdir(parents=True, exist_ok=True)

    state = {"finalized": False}
    _record_run_lifecycle_event(
        lifecycle_path=lifecycle_path,
        status_path=status_path,
        event="start",
        run_name=run_name,
        extra={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        },
    )

    original_excepthook = sys.excepthook

    def _finalize(event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if state["finalized"] and event != "completed":
            return
        if event != "completed":
            state["finalized"] = True
        _record_run_lifecycle_event(
            lifecycle_path=lifecycle_path,
            status_path=status_path,
            event=event,
            run_name=run_name,
            extra=extra,
        )

    def _handle_signal(signum: int, _frame: Any) -> None:
        signal_name = ""
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = f"SIG{signum}"
        _finalize(
            "signal",
            {
                "signal_number": int(signum),
                "signal_name": signal_name,
            },
        )
        raise SystemExit(128 + int(signum))

    def _handle_uncaught_exception(exc_type: Any, exc: BaseException, tb: Any) -> None:
        _finalize(
            "uncaught_exception",
            {
                "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                "error": str(exc or ""),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            },
        )
        original_excepthook(exc_type, exc, tb)

    for signal_name in ("SIGHUP", "SIGTERM", "SIGINT"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _handle_signal)
    sys.excepthook = _handle_uncaught_exception

    return {
        "lifecycle_path": lifecycle_path,
        "status_path": status_path,
        "record_completed": lambda extra=None: _finalize("completed", extra),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated QUALITY benchmark experiments.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--manifest", default="experiments/quality/artifacts/split_manifest.json")
    parser.add_argument("--run-name", default=f"quality_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_SETTING_GROUPS.keys()), default="all")
    parser.add_argument("--settings", default="")
    parser.add_argument("--offline-runtime-source-dir", default="")
    parser.add_argument("--workspace-source-root", default="")
    parser.add_argument("--workspace-asset-root", default="experiments/quality/assets/article_workspaces")
    parser.add_argument("--skip-completed-settings", action="store_true")
    parser.add_argument("--rebuild-articles", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--setting-repeats", default="")
    parser.add_argument("--eval-max-workers", type=int, default=4)
    parser.add_argument("--offline-eval-max-workers", type=int, default=5)
    parser.add_argument("--offline-training-max-workers", type=int, default=1)
    parser.add_argument("--max-parallel-settings", type=int, default=0)
    parser.add_argument("--online-warmup-questions", type=int, default=0)
    parser.add_argument("--online-batch-size", type=int, default=3)
    parser.add_argument("--online-attempts-per-question", type=int, default=1)
    parser.add_argument("--skip-offline-training", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument("--disable-sql-tools", action="store_true")
    parser.add_argument("--reuse-existing-offline-runtime", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.train_only and not str(args.settings or "").strip():
        selected_settings = []
    else:
        selected_settings = _resolve_selected_settings(experiment=args.experiment, settings_arg=args.settings)
    setting_repeats = _resolve_setting_repeats(
        selected_settings=selected_settings,
        global_repeats=args.repeats,
        setting_repeats_arg=args.setting_repeats,
    )

    base_cfg = KAGConfig.from_yaml(args.config)
    _set_global_language(base_cfg, "en")
    base_cfg.global_.doc_type = "general"
    base_cfg.global_config.doc_type = "general"
    base_cfg.global_.aggregation_mode = "narrative"
    base_cfg.global_config.aggregation_mode = "narrative"
    enable_sql_tools = not bool(args.disable_sql_tools)

    manifest = _load_manifest((REPO_ROOT / args.manifest).resolve())
    train_rows = list(manifest.get("train") or [])
    eval_rows = list(manifest.get("eval") or [])
    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
    if args.eval_limit > 0:
        eval_rows = eval_rows[: args.eval_limit]
    if args.smoke:
        train_rows = train_rows[:1]
        eval_rows = eval_rows[:1]

    run_root = REPO_ROOT / "experiments" / "quality" / "runs" / args.run_name
    workspace_root = run_root / "article_workspaces"
    data_root = run_root / "data"
    report_root = run_root / "reports"
    runtime_root = run_root / "runtime"
    for path in (workspace_root, data_root, report_root, runtime_root):
        path.mkdir(parents=True, exist_ok=True)
    lifecycle = _install_run_lifecycle_logging(run_root=run_root, run_name=args.run_name)

    offline_runtime_library = runtime_root / "offline" / "strategy_library.json"
    offline_runtime_tool_meta = runtime_root / "offline" / "tool_metadata"
    offline_runtime_source_dir = (
        _resolve_cli_path(args.offline_runtime_source_dir)
        if str(args.offline_runtime_source_dir or "").strip()
        else None
    )
    workspace_source_root = (
        _resolve_cli_path(args.workspace_source_root)
        if str(args.workspace_source_root or "").strip()
        else None
    )
    workspace_asset_root = (
        _resolve_cli_path(args.workspace_asset_root)
        if str(args.workspace_asset_root or "").strip()
        else _default_workspace_asset_root().resolve()
    )
    workspace_asset_root.mkdir(parents=True, exist_ok=True)
    if offline_runtime_source_dir is not None:
        if not args.skip_offline_training:
            raise ValueError("--offline-runtime-source-dir requires --skip-offline-training")
        _materialize_external_offline_runtime(
            source_dir=offline_runtime_source_dir,
            runtime_library_path=offline_runtime_library,
            runtime_tool_metadata_dir=offline_runtime_tool_meta,
        )

    offline_accumulator = StrategyLibraryAccumulator(
        cfg=base_cfg,
        runtime_library_path=offline_runtime_library,
        runtime_tool_metadata_dir=offline_runtime_tool_meta,
        output_dir=report_root / "offline_library_build",
        dataset_name=f"{args.run_name}_offline",
    )
    online_repeat_states = _build_online_repeat_states(
        base_cfg=base_cfg,
        runtime_root=runtime_root,
        report_root=report_root,
        run_name=args.run_name,
        selected_settings=selected_settings,
        setting_repeats=setting_repeats,
    )
    reuse_offline_runtime = bool(
        args.reuse_existing_offline_runtime
        and args.skip_offline_training
        and offline_runtime_library.exists()
    )
    if not reuse_offline_runtime:
        offline_accumulator.clear_runtime()
    _reset_online_repeat_states(
        online_repeat_states,
        clear_checkpoints=not bool(args.skip_completed_settings),
    )

    runtime_backup_root = run_root / "runtime_backups"
    original_runtime_library = REPO_ROOT / str(base_cfg.strategy_memory.library_path)
    original_runtime_tool_meta = REPO_ROOT / str(base_cfg.strategy_memory.tool_metadata_runtime_dir)
    if original_runtime_library.exists():
        _copy_runtime_backup(original_runtime_library, runtime_backup_root / "strategy_library.json")
    if original_runtime_tool_meta.exists():
        if (runtime_backup_root / "tool_metadata").exists():
            shutil.rmtree(runtime_backup_root / "tool_metadata")
        shutil.copytree(original_runtime_tool_meta, runtime_backup_root / "tool_metadata")

    offline_training_reports: List[Dict[str, Any]] = []
    if not args.skip_offline_training:
        offline_training_workers = max(
            1,
            min(
                int(args.offline_training_max_workers or 1),
                len(train_rows) or 1,
            ),
        )

        def _run_single_offline_training(article: Dict[str, Any]) -> Dict[str, Any]:
            article_name = str(article.get("article_name", "") or "")
            article_json_path = Path(str(article.get("converted_path", "") or ""))
            article_qa_path = Path(str(article.get("qa_path", "") or ""))
            article_run_root = run_root / "offline_training" / "article_runs" / article_name
            workspace_dir = _resolve_article_workspace_dir(
                article_name=article_name,
                workspace_root=workspace_root,
                workspace_source_root=workspace_source_root,
                workspace_asset_root=workspace_asset_root,
            )
            training_csv = data_root / "offline_training_csvs" / f"{article_name}.csv"
            article_runtime_library = run_root / "offline_training" / article_name / "runtime" / "strategy_library.json"
            article_runtime_tool_meta = run_root / "offline_training" / article_name / "runtime" / "tool_metadata"
            existing_result_paths = {
                "manifest_path": article_run_root / "manifests" / "training_manifest.json",
                "question_summary_path": article_run_root / "distilled" / "question_training_summaries.json",
                "raw_template_path": article_run_root / "distilled" / "raw_templates.json",
                "failure_summary_path": article_run_root / "failures" / "failed_question_summaries.json",
                "reflection_path": article_run_root / "failures" / "attempt_reflections.jsonl",
                "retry_attempts_path": article_run_root / "attempts" / "retry_attempts.jsonl",
                "tool_reflection_record_path": article_run_root / "tool_metadata" / "tool_description_reflection_records.jsonl",
                "tool_description_candidate_path": article_run_root / "tool_metadata" / "tool_description_candidates.json",
                "cluster_path": article_run_root / "clusters" / "template_clusters.json",
                "merge_decision_path": article_run_root / "clusters" / "merge_decisions.jsonl",
                "library_output_path": article_run_root / "library" / "strategy_library.json",
                "template_source_index_path": article_run_root / "library" / "template_source_index.json",
                "runtime_library_path": article_runtime_library,
                "runtime_template_source_index_path": article_runtime_library.parent / "template_source_index.json",
                "report_path": article_run_root / "report.md",
            }
            existing_required = [
                existing_result_paths["raw_template_path"],
                existing_result_paths["tool_description_candidate_path"],
                existing_result_paths["library_output_path"],
                existing_result_paths["runtime_library_path"],
                existing_result_paths["report_path"],
            ]
            if all(path.exists() for path in existing_required):
                question_summaries = load_json(str(existing_result_paths["question_summary_path"]))
                failed_summaries = load_json(str(existing_result_paths["failure_summary_path"]))
                tool_candidates = load_json(str(existing_result_paths["tool_description_candidate_path"]))
                runtime_tool_metadata_paths = (
                    sorted(str(path) for path in article_runtime_tool_meta.rglob("*.json"))
                    if article_runtime_tool_meta.exists()
                    else []
                )
                return {
                    "article_name": article_name,
                    "training_csv": str(training_csv),
                    **{key: str(path) for key, path in existing_result_paths.items()},
                    "runtime_tool_metadata_dir": str(article_runtime_tool_meta),
                    "runtime_tool_metadata_paths": runtime_tool_metadata_paths,
                    "successful_question_count": len(question_summaries if isinstance(question_summaries, list) else []),
                    "failed_question_count": len(failed_summaries if isinstance(failed_summaries, list) else []),
                    "tool_description_candidate_count": len(tool_candidates if isinstance(tool_candidates, list) else []),
                    "resumed_from_existing": True,
                }
            _prepare_article_cfg(
                base_cfg=base_cfg,
                article_name=article_name,
                article_json_path=article_json_path,
                workspace_dir=workspace_dir,
                workspace_asset_root=workspace_asset_root,
                rebuild=args.rebuild_articles,
                reuse_existing_workspace_only=workspace_source_root is not None,
            )
            qa_rows = _load_article_qas(article_qa_path, with_answer=True)
            training_csv = _write_offline_training_csv(training_csv, qa_rows)
            train_cfg = _build_article_config(
                base_cfg,
                workspace_dir=workspace_dir,
                runtime_library_path=article_runtime_library,
                runtime_tool_metadata_dir=article_runtime_tool_meta,
                strategy_enabled=False,
                subagent_enabled=False,
            )
            train_cfg.strategy_memory.training_max_workers = max(
                1,
                min(
                    int(getattr(train_cfg.strategy_memory, "training_max_workers", 5) or 5),
                    len(qa_rows),
                ),
            )
            runner = StrategyMemoryTrainingRunner(
                config=train_cfg,
                csv_path=str(training_csv),
                dataset_name=article_name,
                attempts_per_question=5,
                output_root=str(run_root.relative_to(REPO_ROOT) / "offline_training" / "article_runs"),
                runtime_library_path=str(article_runtime_library),
                enable_sql_tools=enable_sql_tools,
            )
            try:
                result = runner.run(reset_runtime_library=True)
            finally:
                runner.close()
            return {
                "article_name": article_name,
                "training_csv": str(training_csv),
                **result,
            }

        completed_training_reports: Dict[str, Dict[str, Any]] = {}
        if offline_training_workers <= 1:
            for article in train_rows:
                report = _run_single_offline_training(article)
                completed_training_reports[str(report.get("article_name", "") or "")] = report
        else:
            with ThreadPoolExecutor(max_workers=offline_training_workers) as executor:
                future_to_article_name = {
                    executor.submit(_run_single_offline_training, article): str(article.get("article_name", "") or "")
                    for article in train_rows
                }
                for future in as_completed(future_to_article_name):
                    report = future.result()
                    completed_training_reports[str(report.get("article_name", "") or "")] = report

        for article in train_rows:
            article_name = str(article.get("article_name", "") or "")
            report = completed_training_reports.get(article_name)
            if not isinstance(report, dict):
                raise RuntimeError(f"Missing offline training report for article: {article_name}")
            raw_templates = load_json(report["raw_template_path"])
            tool_candidates = load_json(report["tool_description_candidate_path"])
            offline_accumulator.add_templates(raw_templates if isinstance(raw_templates, list) else [], template_prefix=article_name)
            offline_accumulator.add_tool_candidates(tool_candidates if isinstance(tool_candidates, list) else [])
            offline_training_reports.append(report)
        offline_library_summary = offline_accumulator.export(consolidate=True, export_name="final")
    else:
        offline_library_summary = _load_existing_runtime_summary(
            runtime_library_path=offline_runtime_library,
            runtime_tool_metadata_dir=offline_runtime_tool_meta,
            reused=reuse_offline_runtime,
        )

    setting_results: Dict[str, List[Dict[str, Any]]] = _empty_setting_results(selected_settings)
    online_training_reports: List[Dict[str, Any]] = []
    parallel_setting_workers = max(
        1,
        min(
            int(args.max_parallel_settings or len(selected_settings) or 1),
            len(selected_settings) or 1,
        ),
    )

    if not args.train_only:
        for article in eval_rows:
            article_name = str(article.get("article_name", "") or "")
            article_json_path = Path(str(article.get("converted_path", "") or ""))
            article_qa_path = Path(str(article.get("qa_path", "") or ""))
            workspace_dir = _resolve_article_workspace_dir(
                article_name=article_name,
                workspace_root=workspace_root,
                workspace_source_root=workspace_source_root,
                workspace_asset_root=workspace_asset_root,
            )
            qa_rows = _load_article_qas(article_qa_path, with_answer=True)
            offline_eval_workers = max(1, min(int(args.offline_eval_max_workers or 1), len(qa_rows) or 1))
            article_setting_results = _empty_setting_results(selected_settings)
            article_online_training_reports: List[Dict[str, Any]] = []
            _prepare_article_cfg(
                base_cfg=base_cfg,
                article_name=article_name,
                article_json_path=article_json_path,
                workspace_dir=workspace_dir,
                workspace_asset_root=workspace_asset_root,
                rebuild=args.rebuild_articles,
                reuse_existing_workspace_only=workspace_source_root is not None,
            )

            article_progress_lock = threading.Lock()

            def _persist_completed_setting(
                *,
                setting_name: str,
                results: List[Dict[str, Any]],
                training_reports: Optional[List[Dict[str, Any]]] = None,
            ) -> None:
                with article_progress_lock:
                    article_setting_results[setting_name] = list(results)
                    if training_reports:
                        article_online_training_reports.extend(list(training_reports))
                    _persist_setting_progress(
                        report_root=report_root,
                        article_name=article_name,
                        setting_name=setting_name,
                        results=article_setting_results[setting_name],
                        article_setting_results=article_setting_results,
                        online_training_reports=article_online_training_reports,
                        setting_repeats=setting_repeats,
                        setting_order=selected_settings,
                    )

            def _run_single_setting(setting_name: str) -> None:
                existing_payload = _load_existing_setting_payload(report_root, article_name, setting_name)
                expected_repeats = int(setting_repeats.get(setting_name, 1) or 1)
                if args.skip_completed_settings and _setting_payload_is_complete(
                    existing_payload,
                    expected_question_count=len(qa_rows),
                    repeats=expected_repeats,
                ):
                    if setting_name in online_repeat_states:
                        repeat_states = online_repeat_states.get(setting_name, [])
                        if len(repeat_states) != expected_repeats:
                            raise ValueError(
                                f"Online repeat state mismatch for {setting_name}: expected {expected_repeats}, got {len(repeat_states)}"
                            )
                        for repeat_state in repeat_states:
                            _restore_online_repeat_state_from_checkpoint(repeat_state, article_name=article_name)
                    cached_results = list(existing_payload.get("results") or [])
                    cached_reports = [item for item in (existing_payload.get("online_training_reports") or []) if isinstance(item, dict)]
                    _persist_completed_setting(
                        setting_name=setting_name,
                        results=cached_results,
                        training_reports=cached_reports,
                    )
                    return

                if setting_name == "offline_strategy_agent":
                    cfg = _build_article_config(
                        base_cfg,
                        workspace_dir=workspace_dir,
                        runtime_library_path=offline_runtime_library,
                        runtime_tool_metadata_dir=offline_runtime_tool_meta,
                        strategy_enabled=True,
                        subagent_enabled=False,
                    )
                    evaluator = AgentEvaluator(
                        cfg,
                        setting_name=setting_name,
                        article_name=article_name,
                        enable_sql_tools=enable_sql_tools,
                    )
                    try:
                        results = _evaluate_setting(
                            rows=qa_rows,
                            evaluator=evaluator,
                            repeats=expected_repeats,
                            max_workers=offline_eval_workers,
                        )
                    finally:
                        evaluator.close()
                    _persist_completed_setting(setting_name=setting_name, results=results)
                    return

                if setting_name == "offline_strategy_subagent":
                    cfg = _build_article_config(
                        base_cfg,
                        workspace_dir=workspace_dir,
                        runtime_library_path=offline_runtime_library,
                        runtime_tool_metadata_dir=offline_runtime_tool_meta,
                        strategy_enabled=True,
                        subagent_enabled=True,
                    )
                    evaluator = AgentEvaluator(
                        cfg,
                        setting_name=setting_name,
                        article_name=article_name,
                        enable_sql_tools=enable_sql_tools,
                    )
                    try:
                        results = _evaluate_setting(
                            rows=qa_rows,
                            evaluator=evaluator,
                            repeats=expected_repeats,
                            max_workers=offline_eval_workers,
                        )
                    finally:
                        evaluator.close()
                    _persist_completed_setting(setting_name=setting_name, results=results)
                    return

                if setting_name == "no_strategy_agent":
                    cfg = _build_article_config(
                        base_cfg,
                        workspace_dir=workspace_dir,
                        strategy_enabled=False,
                        subagent_enabled=False,
                        strategy_read_enabled=False,
                        runtime_routing_note_enabled=False,
                    )
                    evaluator = AgentEvaluator(
                        cfg,
                        setting_name=setting_name,
                        article_name=article_name,
                        enable_sql_tools=enable_sql_tools,
                    )
                    try:
                        results = _evaluate_setting(
                            rows=qa_rows,
                            evaluator=evaluator,
                            repeats=expected_repeats,
                            max_workers=offline_eval_workers,
                        )
                    finally:
                        evaluator.close()
                    _persist_completed_setting(setting_name=setting_name, results=results)
                    return

                if setting_name == "traditional_hybrid_rag_bm25":
                    cfg = _build_article_config(
                        base_cfg,
                        workspace_dir=workspace_dir,
                        strategy_enabled=False,
                        subagent_enabled=False,
                        strategy_read_enabled=False,
                        runtime_routing_note_enabled=False,
                    )
                    evaluator = TraditionalHybridEvaluator(cfg, article_name=article_name)
                    results = _evaluate_setting(
                        rows=qa_rows,
                        evaluator=evaluator,
                        repeats=expected_repeats,
                        max_workers=offline_eval_workers,
                    )
                    _persist_completed_setting(setting_name=setting_name, results=results)
                    return

                if setting_name == "online_strategy_agent":
                    if args.skip_online:
                        return
                    repeat_states = online_repeat_states.get(setting_name, [])
                    if len(repeat_states) != expected_repeats:
                        raise ValueError(
                            f"Online repeat state mismatch for {setting_name}: expected {expected_repeats}, got {len(repeat_states)}"
                        )
                    results, reports = _evaluate_online_setting_incremental(
                        base_cfg=base_cfg,
                        run_root=run_root,
                        report_root=report_root,
                        workspace_dir=workspace_dir,
                        article_name=article_name,
                        qa_rows=qa_rows,
                        setting_name=setting_name,
                        subagent_enabled=False,
                        online_repeat_states=repeat_states,
                        self_bootstrap_max_questions=1,
                        warmup_questions=args.online_warmup_questions,
                        batch_size=args.online_batch_size,
                        online_attempts_per_question=args.online_attempts_per_question,
                        eval_max_workers=args.eval_max_workers,
                        enable_sql_tools=enable_sql_tools,
                    )
                    _persist_completed_setting(setting_name=setting_name, results=results, training_reports=reports)
                    return

                if setting_name == "online_strategy_subagent":
                    if args.skip_online:
                        return
                    repeat_states = online_repeat_states.get(setting_name, [])
                    if len(repeat_states) != expected_repeats:
                        raise ValueError(
                            f"Online repeat state mismatch for {setting_name}: expected {expected_repeats}, got {len(repeat_states)}"
                        )
                    results, reports = _evaluate_online_setting_incremental(
                        base_cfg=base_cfg,
                        run_root=run_root,
                        report_root=report_root,
                        workspace_dir=workspace_dir,
                        article_name=article_name,
                        qa_rows=qa_rows,
                        setting_name=setting_name,
                        subagent_enabled=True,
                        online_repeat_states=repeat_states,
                        self_bootstrap_max_questions=1,
                        warmup_questions=args.online_warmup_questions,
                        batch_size=args.online_batch_size,
                        online_attempts_per_question=args.online_attempts_per_question,
                        eval_max_workers=args.eval_max_workers,
                        enable_sql_tools=enable_sql_tools,
                    )
                    _persist_completed_setting(setting_name=setting_name, results=results, training_reports=reports)
                    return

                raise ValueError(f"Unsupported setting: {setting_name}")

            with ThreadPoolExecutor(max_workers=parallel_setting_workers) as executor:
                futures = [executor.submit(_run_single_setting, setting_name) for setting_name in selected_settings]
                for future in as_completed(futures):
                    future.result()

            _flush_article_progress(
                report_root=report_root,
                article_name=article_name,
                setting_results=article_setting_results,
                online_training_reports=article_online_training_reports,
                setting_repeats=setting_repeats,
                setting_order=selected_settings,
            )
            for setting in selected_settings:
                setting_results[setting].extend(article_setting_results.get(setting, []))
            online_training_reports.extend(article_online_training_reports)

    final_online_summary = _summarize_online_repeat_states(online_repeat_states)
    article_payloads = _load_article_result_payloads(report_root)
    report_setting_order = _resolve_report_setting_order(
        article_payloads=article_payloads,
        current_setting_order=selected_settings,
    )
    report_setting_repeats = _resolve_report_setting_repeats(
        article_payloads=article_payloads,
        current_setting_repeats=setting_repeats,
        setting_order=report_setting_order,
    )
    aggregated = _aggregate_article_payloads(
        article_payloads=article_payloads,
        setting_repeats=report_setting_repeats,
        setting_order=report_setting_order,
    )
    summary = aggregated["summary"]
    article_summary = aggregated["article_summary"]
    setting_results = aggregated["setting_results"]
    online_training_reports = aggregated["online_training_reports"]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "manifest": str((REPO_ROOT / args.manifest).resolve()),
        "run_name": args.run_name,
        "experiment": args.experiment,
        "selected_settings": selected_settings,
        "report_setting_order": report_setting_order,
        "setting_repeats": report_setting_repeats,
        "train_article_count": len(train_rows),
        "eval_article_count": len(eval_rows),
        "repeats": int(args.repeats or 0),
        "enable_sql_tools": enable_sql_tools,
        "offline_library_summary": offline_library_summary,
        "online_library_summary": {
            "mode": "per_setting_isolated",
            "settings": final_online_summary,
        },
        "online_library_summaries": final_online_summary,
        "offline_training_reports": offline_training_reports,
        "online_training_reports": online_training_reports,
        "summary": summary,
        "article_summary": article_summary,
        "results": setting_results,
    }
    json_dump_atomic(str(report_root / "quality_benchmark_results.json"), payload)

    lines: List[str] = [
        "# QUALITY Benchmark Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Run name: `{args.run_name}`",
        f"- Train articles: `{len(train_rows)}`",
        f"- Eval articles: `{len(eval_rows)}`",
        f"- Setting repeats: `{_format_setting_repeats_text(report_setting_repeats, report_setting_order)}`",
        "",
        "## Summary",
        "",
    ]
    for setting in report_setting_order:
        row = summary.get(setting, {})
        lines.extend(
            [
                f"### {setting}",
                "",
                f"- Total attempts: `{row['total_attempts']}`",
                f"- Correct attempts: `{row['correct_attempts']}`",
                f"- Overall accuracy: `{row['overall_accuracy']}`",
                f"- Pass accuracy: `{row['pass_accuracy']}`",
                f"- Pass question count: `{row['pass_question_count']}`",
                f"- Average latency: `{row['avg_latency_ms']} ms`",
                "",
            ]
        )
    lines.extend(["## Article Summary", ""])
    for article_name, row in article_summary.items():
        lines.append(f"### {article_name}")
        lines.append("")
        for setting, item in row.items():
            lines.append(
                f"- {setting}: accuracy=`{item['overall_accuracy']}` pass=`{item['pass_accuracy']}` "
                f"correct=`{item['correct_attempts']}/{item['total_attempts']}`"
            )
        lines.append("")
    (report_root / "quality_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    lifecycle["record_completed"](
        {
            "report_json": str(report_root / "quality_benchmark_results.json"),
            "report_md": str(report_root / "quality_benchmark_report.md"),
        }
    )
    print(json.dumps({"report_json": str(report_root / "quality_benchmark_results.json"), "report_md": str(report_root / "quality_benchmark_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
