from __future__ import annotations

import logging
import os
import re
import threading
import importlib
from typing import Any, Dict, List, Sequence, Tuple


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


def _contextual_description(name: str, label: str, score: float, text: str) -> str:
    label_text = str(label or "").strip() or "entity"
    name_text = str(name or "").strip()
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    context = ""
    if name_text and raw:
        pattern = re.compile(r"[^。！？!?；;\n]{0,90}" + re.escape(name_text) + r"[^。！？!?；;\n]{0,90}")
        match = pattern.search(raw)
        if match:
            context = re.sub(r"\s+", " ", match.group(0)).strip(" ，。；;：:,.")
    if context:
        return context
    return _default_description(label_text, score)


def _is_cuda_device(device: str) -> bool:
    text = str(device or "").strip().lower()
    return text == "cuda" or bool(re.fullmatch(r"cuda:\d+", text))


class ExternalEntityCandidateExtractor:
    def __init__(
        self,
        config: Any,
        llm: Any | None = None,
        *,
        entity_mode: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        high_confidence_threshold: float | None = None,
        max_items: int | None = None,
        enable_direct_type: bool | None = None,
        allow_download: bool | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        kg_cfg = getattr(config, "knowledge_graph_builder", None)
        global_cfg = getattr(config, "global_config", None)

        self.language = str(getattr(global_cfg, "language", "") or getattr(global_cfg, "locale", "") or "en").strip().lower() or "en"
        self.entity_mode = str(
            entity_mode if entity_mode is not None else getattr(kg_cfg, "entity_extraction_mode", "llm")
        ).strip().lower() or "llm"
        self.model_name = str(
            model_name if model_name is not None else getattr(kg_cfg, "ner_model_name", "")
        ).strip()
        self.device = str(
            device if device is not None else getattr(kg_cfg, "ner_device", "auto")
        ).strip().lower()
        self.threshold = float(
            threshold if threshold is not None else getattr(kg_cfg, "ner_threshold", 0.55) or 0.55
        )
        self.high_conf_threshold = float(
            high_confidence_threshold
            if high_confidence_threshold is not None
            else getattr(kg_cfg, "ner_high_confidence_threshold", 0.82) or 0.82
        )
        self.max_items = max(
            1,
            int(max_items if max_items is not None else getattr(kg_cfg, "ner_max_items", 24) or 24),
        )
        self.enable_direct_type = bool(
            enable_direct_type
            if enable_direct_type is not None
            else getattr(kg_cfg, "ner_enable_direct_type", True)
        )
        self.allow_download = bool(
            allow_download
            if allow_download is not None
            else getattr(kg_cfg, "ner_allow_download", False)
        )

        self._model = None
        self._pipeline = None
        self._pipeline_task = ""
        self._active_backend = self._resolve_backend()
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._failed = False
        self._last_error = ""

    def _resolve_backend(self) -> str:
        if self.entity_mode != "ner":
            return "none"
        if self.language.startswith("zh"):
            return "modelscope_uie"
        if self.language.startswith("en"):
            return "modelscope_gliner"
        raise ValueError(f"entity_extraction_mode=ner does not support language={self.language!r}")

    def is_enabled(self) -> bool:
        return self._active_backend != "none"

    def backend_name(self) -> str:
        return self._active_backend

    def _label_specs(self) -> List[Tuple[str, str]]:
        if self.language == "en":
            return [
                ("character", "Character"),
                ("location", "Location"),
                ("object", "Object"),
                ("organization", "Concept"),
                ("concept", "Concept"),
                ("per", "Character"),
                ("loc", "Location"),
                ("org", "Concept"),
                ("misc", "Concept"),
            ]
        if self.language == "zh":
            return [
                ("人物", "Character"),
                ("角色", "Character"),
                ("地点", "Location"),
                ("场所", "Location"),
                ("地理位置", "Location"),
                ("道具", "Object"),
                ("物体", "Object"),
                ("物品", "Object"),
                ("设备", "Object"),
                ("组织", "Concept"),
                ("组织机构", "Concept"),
                ("机构", "Concept"),
                ("概念", "Concept"),
                ("系统", "Concept"),
            ]
        return [
            ("character", "Character"),
            ("location", "Location"),
            ("object", "Object"),
            ("organization", "Concept"),
            ("per", "Character"),
            ("loc", "Location"),
            ("org", "Concept"),
        ]

    def _modelscope_device(self) -> str:
        if self.device in {"", "auto", "cpu"}:
            return "cpu"
        if self.device == "cuda":
            return "gpu"
        m = re.fullmatch(r"cuda:(\d+)", self.device)
        if m:
            return f"gpu:{m.group(1)}"
        return self.device

    def _ensure_modelscope_uie(self):
        try:
            if self._failed:
                raise RuntimeError(self._last_error or "ModelScope UIE backend is disabled after a previous load failure.")
            if self._pipeline is not None:
                return self._pipeline
            with self._load_lock:
                if self._failed:
                    raise RuntimeError(self._last_error or "ModelScope UIE backend is disabled after a previous load failure.")
                if self._pipeline is not None:
                    return self._pipeline

                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                modelscope_plugins = importlib.import_module("modelscope.utils.plugins")

                model_name = self.model_name or "iic/nlp_structbert_siamese-uie_chinese-base"
                if model_name.startswith("/") or os.path.isdir(model_name):
                    model_ref = model_name
                else:
                    cache_dir = os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_name}")
                    if not os.path.isdir(cache_dir) and not self.allow_download:
                        raise FileNotFoundError(f"ModelScope NER model cache not found for {model_name}")
                    model_ref = cache_dir if os.path.isdir(cache_dir) else model_name

                task = Tasks.siamese_uie
                original_install_requirements = modelscope_plugins.install_requirements_by_files
                modelscope_plugins.install_requirements_by_files = lambda requirements: None
                try:
                    self._pipeline = pipeline(task, model=model_ref, device=self._modelscope_device())
                finally:
                    modelscope_plugins.install_requirements_by_files = original_install_requirements
                self._pipeline_task = str(task)
                self._last_error = ""
                logger.info("ModelScope UIE loaded from %s", model_ref)
                return self._pipeline
        except Exception as exc:
            self._failed = True
            self._active_backend = "none"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _resolve_modelscope_gliner_path(self) -> str:
        model_name = self.model_name or "knowledgator/gliner-relex-multi-v1___0"
        if model_name.startswith("/") or os.path.isdir(model_name):
            if not os.path.isdir(model_name):
                raise FileNotFoundError(f"ModelScope GLiNER model path not found: {model_name}")
            return model_name
        candidates = [
            os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_name}"),
            os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_name.replace('.', '___')}"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        if not self.allow_download:
            raise FileNotFoundError(f"ModelScope GLiNER model cache not found for {model_name}")
        return model_name

    def _ensure_modelscope_gliner(self):
        try:
            if self._failed:
                raise RuntimeError(self._last_error or "ModelScope GLiNER backend is disabled after a previous load failure.")
            if self._model is not None:
                return self._model
            with self._load_lock:
                if self._failed:
                    raise RuntimeError(self._last_error or "ModelScope GLiNER backend is disabled after a previous load failure.")
                if self._model is not None:
                    return self._model

                from gliner import GLiNER

                model_ref = self._resolve_modelscope_gliner_path()
                self._model = GLiNER.from_pretrained(model_ref, local_files_only=not self.allow_download)
                if _is_cuda_device(self.device):
                    try:
                        self._model.to(self.device)
                    except Exception as exc:
                        logger.warning("GLiNER cuda placement failed; falling back to default device: %s", exc)
                self._pipeline_task = "gliner"
                self._last_error = ""
                logger.info("ModelScope GLiNER loaded from %s", model_ref)
                return self._model
        except Exception as exc:
            self._failed = True
            self._active_backend = "none"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _split_for_modelscope_gliner(self, text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        max_chars = 1200
        if len(raw) <= max_chars:
            return [raw]
        parts = re.split(r"(?<=[.!?;\n])\s+", raw)
        out: List[str] = []
        buf = ""
        for part in parts:
            if not part:
                continue
            if buf and len(buf) + 1 + len(part) > max_chars:
                out.append(buf)
                buf = part
            else:
                buf = f"{buf} {part}".strip()
        if buf:
            out.append(buf)
        return out or [raw[:max_chars]]

    def _extract_modelscope_gliner(self, text: str) -> List[Dict[str, Any]]:
        try:
            ner = self._ensure_modelscope_gliner()
        except Exception as exc:
            if not self._last_error:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"ModelScope GLiNER extraction failed: {self._last_error}") from exc

        labels = [label for label, _etype in self._label_specs() if label in {"character", "location", "object", "concept", "organization"}]
        out: List[Dict[str, Any]] = []
        with self._lock:
            for piece in self._split_for_modelscope_gliner(text):
                result = ner.predict_entities(piece, labels, threshold=self.threshold)
                rows: List[Dict[str, Any]] = []

                def _flatten(value: Any) -> None:
                    if isinstance(value, dict):
                        rows.append(value)
                    elif isinstance(value, list):
                        for sub in value:
                            _flatten(sub)

                _flatten(result)
                for row in rows:
                    span = str(row.get("text") or row.get("span") or "").strip()
                    label = str(row.get("label") or row.get("type") or "").strip()
                    try:
                        score = float(row.get("score", row.get("prob", 1.0)) or 1.0)
                    except Exception:
                        score = 1.0
                    if span and label:
                        out.append({"text": span, "label": label, "score": score})
        return out

    def _split_for_modelscope_uie(self, text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        max_chars = 450
        if len(raw) <= max_chars:
            return [raw]
        parts = re.split(r"(?<=[。！？!?；;\n])", raw)
        out: List[str] = []
        buf = ""
        for part in parts:
            if not part:
                continue
            if buf and len(buf) + len(part) > max_chars:
                out.append(buf)
                buf = part
            else:
                buf += part
        if buf:
            out.append(buf)
        return out or [raw[:max_chars]]

    def _extract_modelscope_uie(self, text: str) -> List[Dict[str, Any]]:
        try:
            ner = self._ensure_modelscope_uie()
        except Exception as exc:
            if not self._last_error:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"ModelScope UIE extraction failed: {self._last_error}") from exc

        out: List[Dict[str, Any]] = []
        with self._lock:
            for piece in self._split_for_modelscope_uie(text):
                result = ner(
                    input=piece,
                    schema={
                        "人物": None,
                        "地点": None,
                        "物体": None,
                        "道具": None,
                        "设备": None,
                        "组织机构": None,
                        "概念": None,
                        "系统": None,
                    },
                )
                rows = []
                def _flatten(value: Any, inherited_label: str = "") -> None:
                    if isinstance(value, dict):
                        row = dict(value)
                        if inherited_label and not row.get("type"):
                            row["type"] = inherited_label
                        rows.append(row)
                    elif isinstance(value, list):
                        for sub in value:
                            _flatten(sub, inherited_label)
                    elif str(value).strip():
                        rows.append({"span": str(value).strip(), "type": inherited_label, "prob": 1.0})

                if isinstance(result, dict):
                    for label, values in result.items():
                        _flatten(values, "" if label == "output" else label)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    span = str(row.get("span") or row.get("word") or "").strip()
                    label = str(row.get("type") or row.get("entity_group") or row.get("entity") or "").strip()
                    try:
                        score = float(row.get("prob", row.get("score", 1.0)) or 1.0)
                    except Exception:
                        score = 1.0
                    if span and label:
                        out.append({"text": span, "label": label, "score": score})
        return out

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

        raw_items: List[Dict[str, Any]]
        if self._active_backend == "modelscope_uie":
            raw_items = self._extract_modelscope_uie(text)
        elif self._active_backend == "modelscope_gliner":
            raw_items = self._extract_modelscope_gliner(text)
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
            description = _contextual_description(name, label, score, text)

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
