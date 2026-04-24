from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Sequence, Tuple

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template, general_rules
from core.utils.general_utils import load_json


logger = logging.getLogger(__name__)


_SCREENPLAY_HEADER_RE = re.compile(r"^(?:INT|EXT|INT/EXT|EXT/INT|CUT TO|FADE IN|FADE OUT|DISSOLVE TO)[:.\s-]*$", re.IGNORECASE)
_ALNUM_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _norm_name(text: Any) -> str:
    return str(text or "").strip().lower()


def _dedup_open_entities(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
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


def _dedup_typed_entities(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not name or not etype:
            continue
        key = (name.lower(), etype)
        cur = merged.get(key)
        if cur is None or len(desc) > len(str(cur.get("description", ""))):
            merged[key] = dict(item)
    return list(merged.values())


def _looks_useful_name(name: str) -> bool:
    raw = str(name or "").strip()
    if not raw:
        return False
    if not _ALNUM_RE.search(raw):
        return False
    if _SCREENPLAY_HEADER_RE.match(raw):
        return False
    words = raw.split()
    if len(words) > 8:
        return False
    return True


def _default_description(label: str, score: float) -> str:
    label_text = str(label or "").strip() or "entity"
    return f"Detected by external NER as {label_text} (confidence={score:.2f}) in the current passage."


def _is_cuda_device(device: str) -> bool:
    text = str(device or "").strip().lower()
    return text == "cuda" or bool(re.fullmatch(r"cuda:\d+", text))


class ExternalEntityCandidateExtractor:
    def __init__(self, config: Any, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm
        kg_cfg = getattr(config, "knowledge_graph_builder", None)
        global_cfg = getattr(config, "global_config", None)

        self.language = str(getattr(global_cfg, "language", "") or getattr(global_cfg, "locale", "") or "en").strip().lower() or "en"
        self.backend = str(getattr(kg_cfg, "fast_external_entity_backend", "none") or "none").strip().lower()
        self.model_name = str(getattr(kg_cfg, "fast_external_entity_model_name", "") or "").strip()
        self.device = str(getattr(kg_cfg, "fast_external_entity_device", "auto") or "auto").strip().lower()
        self.threshold = float(getattr(kg_cfg, "fast_external_entity_threshold", 0.55) or 0.55)
        self.high_conf_threshold = float(getattr(kg_cfg, "fast_external_entity_high_confidence_threshold", 0.82) or 0.82)
        self.max_items = max(1, int(getattr(kg_cfg, "fast_external_entity_max_items", 24) or 24))
        self.enable_direct_type = bool(getattr(kg_cfg, "fast_external_entity_enable_direct_type", True))
        self.qwen_max_retries = 1

        self._model = None
        self._active_backend = self._resolve_backend()
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._failed = False
        self._last_error = ""
        self._entity_task_block = self._load_entity_task_block()

    def _resolve_backend(self) -> str:
        if self.backend in {"none", ""}:
            return "none"
        if self.backend == "auto":
            if self.llm is not None:
                return "qwen"
            if self.language == "en":
                try:
                    import gliner  # noqa: F401
                    return "gliner"
                except Exception:
                    return "none"
            if self.language == "zh":
                try:
                    import paddlenlp  # noqa: F401
                    import paddle  # noqa: F401
                    return "uie"
                except Exception:
                    return "none"
            return "none"
        return self.backend

    def _load_entity_task_block(self) -> Dict[str, Any]:
        global_cfg = getattr(self.config, "global_config", None)
        task_dir = str(getattr(global_cfg, "task_dir", "") or "").strip()
        if not task_dir:
            return {}
        path = f"{task_dir}/entity_extraction_task.json"
        try:
            payload = load_json(path)
        except Exception:
            return {}
        if not isinstance(payload, list):
            return {}
        for item in payload:
            if isinstance(item, dict) and str(item.get("task", "")).strip() == "general_entity_extraction":
                return item
        return {}

    def is_enabled(self) -> bool:
        return self._active_backend != "none"

    def backend_name(self) -> str:
        return self._active_backend

    def _ensure_gliner(self):
        try:
            if self._failed:
                raise RuntimeError(self._last_error or "GLiNER backend is disabled after a previous load failure.")
            if self._model is not None:
                return self._model
            with self._load_lock:
                if self._failed:
                    raise RuntimeError(self._last_error or "GLiNER backend is disabled after a previous load failure.")
                if self._model is not None:
                    return self._model

                from gliner import GLiNER
                import torch

                model_name = self.model_name or "urchade/gliner_small-v2.1"
                is_local_path = bool(model_name.startswith("/")) or bool(re.match(r"^[A-Za-z]:[\\\\/]", model_name))
                if not is_local_path:
                    cache_key = model_name.replace("/", "--")
                    cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{cache_key}")
                    if not os.path.isdir(cache_dir):
                        raise FileNotFoundError(f"GLiNER model cache not found for {model_name}")

                def _choose_device() -> str:
                    if self.device == "cpu":
                        return self.device
                    if _is_cuda_device(self.device):
                        return self.device
                    if not torch.cuda.is_available():
                        return "cpu"
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                        if free_bytes < 2 * 1024 * 1024 * 1024:
                            logger.warning(
                                "GLiNER falling back to CPU because free CUDA memory is low: %.2f GiB / %.2f GiB",
                                free_bytes / (1024 ** 3),
                                total_bytes / (1024 ** 3),
                            )
                            return "cpu"
                    except Exception:
                        pass
                    return "cuda"

                preferred_device = _choose_device()
                candidate_devices = [preferred_device]
                if _is_cuda_device(preferred_device):
                    candidate_devices.append("cpu")

                last_exc: Exception | None = None
                for map_location in candidate_devices:
                    load_kwargs = {"map_location": map_location}
                    if not is_local_path:
                        load_kwargs["local_files_only"] = True
                    try:
                        model = GLiNER.from_pretrained(model_name, **load_kwargs)
                        model.eval()
                        self._model = model
                        self._last_error = ""
                        logger.info("GLiNER loaded on %s from %s", map_location, model_name)
                        return self._model
                    except Exception as exc:
                        last_exc = exc
                        text = f"{type(exc).__name__}: {exc}"
                        if _is_cuda_device(map_location) and "out of memory" in text.lower():
                            logger.warning("GLiNER CUDA load OOM, retrying on CPU")
                            continue
                        break
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("GLiNER load failed with no captured exception")
        except Exception as exc:
            self._failed = True
            self._active_backend = "none"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _label_specs(self) -> List[Tuple[str, str]]:
        if self.language == "en":
            return [
                ("character", "Character"),
                ("location", "Location"),
                ("object", "Object"),
                ("organization", "Concept"),
            ]
        if self.language == "zh":
            return [
                ("人物", "Character"),
                ("地点", "Location"),
                ("道具", "Object"),
                ("组织", "Concept"),
            ]
        return [
            ("character", "Character"),
            ("location", "Location"),
            ("object", "Object"),
            ("organization", "Concept"),
        ]

    def _extract_gliner(self, text: str) -> List[Dict[str, Any]]:
        try:
            model = self._ensure_gliner()
        except Exception as exc:
            if not self._last_error:
                self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("GLiNER extraction disabled: %s", self._last_error)
            return []
        labels = [label for label, _ in self._label_specs()]
        with self._lock:
            preds = model.predict_entities(text, labels, threshold=self.threshold)
        if not isinstance(preds, list):
            return []
        return [item for item in preds if isinstance(item, dict)]

    def _qwen_type_guidance(self) -> str:
        task_block = self._entity_task_block or {}
        parts: List[str] = []
        for item in task_block.get("types", []) or []:
            if not isinstance(item, dict):
                continue
            etype = str(item.get("type", "")).strip()
            desc = str(item.get("description", "")).strip()
            exclusions = [str(x).strip() for x in (item.get("exclusions") or []) if str(x).strip()]
            if not etype:
                continue
            line = f"- {etype}: {desc}"
            if exclusions:
                line += f" Exclude: {'; '.join(exclusions[:4])}."
            parts.append(line)
        if parts:
            return "\n".join(parts)
        return "\n".join(
            [
                "- Character: concrete individual person or individualized agent relevant to the story.",
                "- Location: physical place or setting where actions happen.",
                "- Object: concrete physical item with narrative utility.",
                "- Concept: organization, institution, faction, family/group, social structure, or abstract plot-relevant concept.",
            ]
        )

    def _extract_qwen(self, text: str, known_names: Sequence[str] | None = None) -> Dict[str, Any]:
        if self.llm is None:
            return {"typed_entities": [], "open_candidates": [], "stats": {"backend": "none", "raw_count": 0, "typed_count": 0, "open_count": 0}}

        known_lines = "\n".join(sorted({str(x).strip() for x in (known_names or []) if str(x).strip()}))
        system_prompt = (
            "You are an assistant for fast narrative entity extraction.\n"
            "Return JSON only.\n"
            f"{general_rules.strip()}"
        )
        user_prompt = (
            "Extract only high-value narrative entities from the passage.\n"
            "Event, Occasion, and TimePoint are already handled elsewhere, so do NOT extract them here.\n"
            "Prefer entities that matter for plot, social structure, recurring objects, and stable locations.\n"
            "Skip obvious background clutter, generic one-off mentions, and screenplay formatting artifacts.\n"
            "If an entity is useful and the type is clear, assign one of: Character, Location, Object, Concept.\n"
            "If useful but not fully certain, still return it with lower confidence.\n"
            f"Return at most {self.max_items} items.\n\n"
            "Type guidance:\n"
            f"{self._qwen_type_guidance()}\n\n"
            "Already known entities (do not repeat):\n"
            f"{known_lines or '(none)'}\n\n"
            "Output JSON object schema:\n"
            '{\n'
            '  "entities": [\n'
            '    {"name": "string", "type": "Character|Location|Object|Concept|Unknown", "description": "string", "confidence": 0.0}\n'
            "  ]\n"
            "}\n\n"
            "Passage:\n"
            f"{text}"
        )
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            required_fields=[],
            field_validators={},
            max_retries=self.qwen_max_retries,
            repair_template=general_repair_template,
        )
        if status != "success":
            return {"typed_entities": [], "open_candidates": [], "stats": {"backend": "qwen", "raw_count": 0, "typed_count": 0, "open_count": 0}}
        try:
            payload = json.loads(corrected_json)
        except Exception as exc:
            logger.warning("qwen external entity parse failed: %s", exc)
            payload = {}
        items = payload.get("entities") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []

        typed_entities: List[Dict[str, Any]] = []
        open_candidates: List[Dict[str, str]] = []
        known = {_norm_name(x) for x in (known_names or []) if _norm_name(x)}

        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip()
            description = str(item.get("description", "")).strip()
            try:
                score = float(item.get("confidence", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if not _looks_useful_name(name):
                continue
            if _norm_name(name) in known:
                continue
            if not description:
                description = _default_description(etype or "entity", score)
            if self.enable_direct_type and etype in {"Character", "Location", "Object", "Concept"} and score >= self.high_conf_threshold:
                typed_entities.append(
                    {
                        "name": name,
                        "type": etype,
                        "description": description,
                        "scope": "global" if etype == "Concept" else "local",
                        "source_kind": "external_typed",
                        "confidence": round(score, 4),
                    }
                )
                known.add(_norm_name(name))
                continue
            open_candidates.append({"name": name, "description": description})

        typed_entities = _dedup_typed_entities(typed_entities)[: self.max_items]
        open_candidates = _dedup_open_entities(open_candidates)[: self.max_items]
        return {
            "typed_entities": typed_entities,
            "open_candidates": open_candidates,
            "stats": {
                "backend": "qwen",
                "raw_count": len(items),
                "typed_count": len(typed_entities),
                "open_count": len(open_candidates),
            },
        }

    def extract(
        self,
        *,
        text: str,
        known_names: Sequence[str] | None = None,
        scope_rules: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        known = {_norm_name(x) for x in (known_names or []) if _norm_name(x)}
        scope_rules = scope_rules or {}

        if not text or not self.is_enabled():
            stats = {"backend": self._active_backend, "raw_count": 0, "typed_count": 0, "open_count": 0}
            if self._last_error:
                stats["error"] = self._last_error
            return {"typed_entities": [], "open_candidates": [], "stats": stats}

        if self._active_backend == "qwen":
            return self._extract_qwen(text, known_names=known_names)

        raw_items: List[Dict[str, Any]]
        if self._active_backend == "gliner":
            raw_items = self._extract_gliner(text)
        else:
            raw_items = []

        label_to_type = {label.lower(): etype for label, etype in self._label_specs()}
        typed_entities: List[Dict[str, Any]] = []
        open_candidates: List[Dict[str, str]] = []

        for item in raw_items:
            name = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).strip()
            score = float(item.get("score", 0.0) or 0.0)
            if not _looks_useful_name(name):
                continue
            if _norm_name(name) in known:
                continue
            mapped_type = label_to_type.get(label.lower(), "")
            description = _default_description(label, score)

            if self.enable_direct_type and mapped_type and score >= self.high_conf_threshold:
                typed_entities.append(
                    {
                        "name": name,
                        "type": mapped_type,
                        "description": description,
                        "scope": scope_rules.get(mapped_type, "local"),
                        "source_kind": "external_typed",
                        "confidence": round(score, 4),
                    }
                )
                known.add(_norm_name(name))
                continue

            open_candidates.append({"name": name, "description": description})

        typed_entities = _dedup_typed_entities(typed_entities)[: self.max_items]
        open_candidates = _dedup_open_entities(open_candidates)[: self.max_items]
        return {
            "typed_entities": typed_entities,
            "open_candidates": open_candidates,
            "stats": {
                "backend": self._active_backend,
                "raw_count": len(raw_items),
                "typed_count": len(typed_entities),
                "open_count": len(open_candidates),
                "error": self._last_error,
            },
        }
