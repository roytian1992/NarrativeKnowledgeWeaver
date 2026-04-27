import json
import hashlib
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Literal
from itertools import chain

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from core.models.data import Document, TextChunk
from ..utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.functions.regular_functions.external_entity_candidates import ExternalEntityCandidateExtractor
from core.utils.format import correct_json_format, safe_text_for_json
from core.utils.function_manager import run_concurrent_with_retries
from core.utils.general_utils import word_len


class DocumentProcessor:
    """
    Narrative-aware document processor (single-pass design):
      1) Split once with a recursive splitter (base_splitter), then optionally refine via
         Sliding Semantic Split (LLM boundary detection).
      2) Run sliding-window summaries directly on the *post-split* chunks (if enabled).
      3) Compute document-level metadata ONCE from (Title, Aggregated Summary), and
         attach it to each chunk as `doc_metadata` (if enabled).
      4) (New) Extract a document-level timeline by aggregating chunk-level time elements (if enabled).
    """

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def __init__(self, config: KAGConfig, llm, doc_type: str = "screenplay", max_worker: int = 4):
        self.config = config
        self.llm = llm
        self.doc_type = doc_type
        self.chunk_size = config.document_processing.chunk_size
        self.chunk_overlap = config.document_processing.chunk_overlap
        self.max_segments = config.document_processing.max_segments
        self.max_content = config.document_processing.max_content_size
        self.max_workers = config.document_processing.max_workers
        self.prepare_insights_mode = str(
            getattr(config.document_processing, "prepare_insights_mode", "llm") or "llm"
        ).strip().lower()
        self.nlp_summary_max_sentences = max(
            1, int(getattr(config.document_processing, "nlp_summary_max_sentences", 4) or 4)
        )
        self.nlp_metadata_max_keywords = max(
            1, int(getattr(config.document_processing, "nlp_metadata_max_keywords", 16) or 16)
        )
        self.nlp_metadata_max_entities_per_type = max(
            1, int(getattr(config.document_processing, "nlp_metadata_max_entities_per_type", 24) or 24)
        )
        self.metadata_entity_mode = str(
            getattr(config.document_processing, "metadata_entity_mode", "llm") or "llm"
        ).strip().lower()
        if self.metadata_entity_mode not in {"llm", "ner"}:
            self.metadata_entity_mode = "llm"
        global_cfg = getattr(config, "global_config", None)
        self.language = str(
            getattr(global_cfg, "language", "") or getattr(global_cfg, "locale", "") or "en"
        ).strip().lower()

        # Base recursive splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,  length_function=word_len
        )

        # LLM-powered parser utilities
        self.document_parser = DocumentParser(config, llm)
        self.metadata_entity_extractor: Optional[ExternalEntityCandidateExtractor] = None
        kg_cfg = getattr(config, "knowledge_graph_builder", None)
        if self.metadata_entity_mode == "ner":
            self.metadata_entity_extractor = ExternalEntityCandidateExtractor(
                config,
                entity_mode="ner",
                max_items=self.nlp_metadata_max_entities_per_type * 4,
                enable_direct_type=True,
            )

    # ------------------------------------------------------------------ #
    # Cheap NLP summary + metadata
    # ------------------------------------------------------------------ #
    _EN_STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "onto", "there", "their",
        "they", "them", "then", "than", "are", "was", "were", "been", "being", "have", "has",
        "had", "not", "but", "you", "your", "his", "her", "she", "him", "our", "out", "all",
        "can", "could", "would", "should", "will", "just", "over", "under", "after", "before",
    }
    _ZH_STOP_CHARS = set("的一是在了和与及或就都而着过把被让向对从到中上下一这那他她它们我你")

    def _is_zh_text(self, text: str) -> bool:
        if self.language.startswith("zh"):
            return True
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))
        latin_count = len(re.findall(r"[A-Za-z]", str(text or "")))
        return zh_count > latin_count

    def _split_summary_sentences(self, text: str) -> List[str]:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return []
        lines = [re.sub(r"\s+", " ", x).strip() for x in raw.splitlines() if x.strip()]
        out: List[str] = []
        for line in lines:
            parts = re.findall(r".+?[。！？!?；;。.]|.+$", line)
            for part in parts:
                part = re.sub(r"\s+", " ", part).strip()
                if part:
                    out.append(part)
        return out

    def _keyword_tokens(self, text: str) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []
        if self._is_zh_text(raw):
            tokens: List[str] = []
            tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", raw))
            for seq in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
                clean = "".join(ch for ch in seq if ch not in self._ZH_STOP_CHARS)
                if len(clean) >= 2:
                    upper = min(5, len(clean))
                    for n in range(2, upper + 1):
                        tokens.extend(clean[i : i + n] for i in range(0, len(clean) - n + 1))
            return [t for t in tokens if 2 <= len(t) <= 24]
        return [
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", raw)
            if t.lower() not in self._EN_STOPWORDS
        ]

    def _nlp_summary(self, text: str, *, max_sentences: Optional[int] = None) -> str:
        sentences = self._split_summary_sentences(text)
        if not sentences:
            return ""
        max_sentences = max(1, int(max_sentences or self.nlp_summary_max_sentences))
        if len(sentences) <= max_sentences:
            return " ".join(sentences).strip()

        tokenized = [self._keyword_tokens(s) for s in sentences]
        freq = Counter(chain.from_iterable(tokenized))
        scored: List[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            toks = tokenized[idx]
            if not toks:
                score = 0.0
            else:
                score = sum(math.log(1.0 + freq.get(tok, 0)) for tok in toks) / math.sqrt(len(toks))
            if idx == 0:
                score += 0.5
            if re.search(r"[：:]", sent):
                score += 0.15
            scored.append((score, idx, sent))

        selected = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
        selected.sort(key=lambda x: x[1])
        return " ".join(s for _, _, s in selected).strip()

    def _dedup_strings(self, items: List[str], *, limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            val = re.sub(r"\s+", " ", str(item or "")).strip(" \t\r\n，。；;：:,.()（）[]【】")
            if not val:
                continue
            key = val.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(val)
            if len(out) >= limit:
                break
        return out

    def _model_ner_metadata(self, *, title: str, text: str) -> Dict[str, Any]:
        extractor = self.metadata_entity_extractor
        if extractor is None:
            return {"entities_by_type": {}, "ner_entities": [], "backend": "none", "error": ""}

        payload = extractor.extract(text=f"{title}\n{text}".strip(), known_names=[], scope_rules={})
        typed_items = payload.get("typed_entities") or []
        stats = payload.get("stats") or {}
        entities_by_type: Dict[str, List[str]] = {
            "Character": [],
            "Location": [],
            "Object": [],
            "Concept": [],
        }
        ner_entities: List[Dict[str, Any]] = []
        for item in typed_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "")).strip()
            if not name or etype not in entities_by_type:
                continue
            entities_by_type[etype].append(name)
            row = dict(item)
            row["source_kind"] = "metadata_ner"
            row.setdefault("description", f"Detected by metadata NER model as {etype}.")
            ner_entities.append(row)

        return {
            "entities_by_type": {
                key: self._dedup_strings(values, limit=self.nlp_metadata_max_entities_per_type)
                for key, values in entities_by_type.items()
            },
            "ner_entities": ner_entities[: self.nlp_metadata_max_entities_per_type * 4],
            "backend": str(stats.get("backend") or extractor.backend_name()),
            "error": str(stats.get("error") or ""),
        }

    def _inject_model_ner_metadata(self, out: Dict[str, Any], *, title: str, text: str) -> Dict[str, Any]:
        if self.metadata_entity_mode != "ner":
            out["metadata_entity_mode"] = "llm"
            return out
        model_payload = self._model_ner_metadata(title=title, text=text)
        model_entities_by_type = model_payload.get("entities_by_type") or {}
        out.update(
            {
                "characters": model_entities_by_type.get("Character", []),
                "locations": model_entities_by_type.get("Location", []),
                "objects": model_entities_by_type.get("Object", []),
                "concepts": model_entities_by_type.get("Concept", []),
                "ner_entities": list(model_payload.get("ner_entities") or []),
                "metadata_entity_mode": "ner",
                "metadata_ner_backend": model_payload.get("backend"),
            }
        )
        if model_payload.get("error"):
            out["metadata_ner_error"] = str(model_payload.get("error"))[:300]
        return out

    def _nlp_metadata(self, *, text: str, title: str, extract_summary: bool, extract_metadata: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {"prepare_insights_mode": "nlp"}
        if extract_summary:
            summary = self._nlp_summary(text, max_sentences=self.nlp_summary_max_sentences)
            if summary:
                out["summary"] = summary
        if not extract_metadata:
            return out

        token_counts = Counter(self._keyword_tokens(f"{title}\n{text}"))
        keywords = [
            token
            for token, _count in token_counts.most_common(self.nlp_metadata_max_keywords * 3)
            if not re.fullmatch(r"\d+", token)
        ][: self.nlp_metadata_max_keywords]
        source_entities_by_type = {"characters": [], "locations": [], "objects": [], "concepts": []}
        ner_entities: List[Dict[str, Any]] = []
        model_payload: Dict[str, Any] = {"backend": "none", "error": ""}
        if self.metadata_entity_mode == "ner":
            model_payload = self._model_ner_metadata(title=title, text=text)
            model_entities_by_type = model_payload.get("entities_by_type") or {}
            source_entities_by_type = {
                "characters": model_entities_by_type.get("Character", []),
                "locations": model_entities_by_type.get("Location", []),
                "objects": model_entities_by_type.get("Object", []),
                "concepts": model_entities_by_type.get("Concept", []),
            }
            ner_entities = list(model_payload.get("ner_entities") or [])

        out.update(
            {
                "keywords": keywords,
                "characters": source_entities_by_type["characters"],
                "locations": source_entities_by_type["locations"],
                "objects": source_entities_by_type["objects"],
                "concepts": source_entities_by_type["concepts"],
                "ner_entities": ner_entities,
                "metadata_entity_mode": self.metadata_entity_mode,
                "metadata_ner_backend": model_payload.get("backend"),
                "knowledge_entity_extraction_mode": str(
                    getattr(getattr(self.config, "knowledge_graph_builder", None), "entity_extraction_mode", "llm") or "llm"
                ),
            }
        )
        if model_payload.get("error"):
            out["metadata_ner_error"] = str(model_payload.get("error"))[:300]
        return out

    # ------------------------------------------------------------------ #
    # Non-LLM semantic splitter
    # ------------------------------------------------------------------ #
    def _semantic_tokens(self, text: str) -> List[str]:
        text = str(text or "").lower()
        if not text:
            return []
        return re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text)

    def _split_into_semantic_sentences(self, text: str) -> List[str]:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return []

        coarse_parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=[.!?;])\s+", raw)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(80, int(self.chunk_size * 0.5)),
            chunk_overlap=0,
            length_function=word_len,
            separators=["\n\n", "\n", "。", "！", "？", "; ", ". ", "! ", "? ", ", ", "，", " ", ""],
        )

        out: List[str] = []
        for part in coarse_parts:
            part = re.sub(r"\s+", " ", str(part or "")).strip()
            if not part:
                continue
            if word_len(part) <= max(40, int(self.chunk_size * 0.5)):
                out.append(part)
                continue
            out.extend([seg.strip() for seg in splitter.split_text(part) if str(seg or "").strip()])
        return out

    def _boundary_strengths(self, sentences: List[str]) -> List[float]:
        n = len(sentences)
        if n <= 1:
            return []

        tokenized = [self._semantic_tokens(s) for s in sentences]
        doc_freq: Counter[str] = Counter()
        for toks in tokenized:
            for tok in set(toks):
                doc_freq[tok] += 1

        def _window_vector(start: int, end: int) -> Dict[str, float]:
            tf: Counter[str] = Counter()
            for toks in tokenized[start:end]:
                tf.update(toks)
            if not tf:
                return {}
            vec: Dict[str, float] = {}
            for tok, freq in tf.items():
                idf = math.log((1.0 + n) / (1.0 + doc_freq.get(tok, 0))) + 1.0
                vec[tok] = float(freq) * idf
            return vec

        def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
            if not a or not b:
                return 0.0
            dot = 0.0
            for tok, val in a.items():
                dot += val * b.get(tok, 0.0)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            if na <= 1e-9 or nb <= 1e-9:
                return 0.0
            return dot / (na * nb)

        strengths: List[float] = []
        for boundary in range(1, n):
            left_start = max(0, boundary - 3)
            right_end = min(n, boundary + 3)
            left_vec = _window_vector(left_start, boundary)
            right_vec = _window_vector(boundary, right_end)
            sim = _cosine(left_vec, right_vec)
            strengths.append(max(0.0, min(1.0, 1.0 - sim)))
        return strengths

    def _semantic_pack_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        sentence_words = [max(1, word_len(s)) for s in sentences]
        total_words = sum(sentence_words)
        target_words = max(200, int(self.chunk_size))
        min_words = max(140, int(target_words * 0.65))
        max_words = max(min_words + 1, int(target_words * 1.30))

        if total_words <= max_words:
            return [" ".join(sentences).strip()]

        strengths = self._boundary_strengths(sentences)
        out: List[str] = []
        start = 0
        n = len(sentences)

        while start < n:
            remaining_words = sum(sentence_words[start:])
            if remaining_words <= max_words:
                out.append(" ".join(sentences[start:]).strip())
                break

            best_end: Optional[int] = None
            best_score = -1e18
            chunk_words = 0
            candidate_ends: List[int] = []

            for end in range(start + 1, n):
                chunk_words += sentence_words[end - 1]
                if chunk_words < min_words:
                    continue
                if chunk_words > max_words:
                    break

                tail_words = sum(sentence_words[end:])
                if tail_words and tail_words < min_words and end < n - 1:
                    continue
                candidate_ends.append(end)

            if not candidate_ends:
                chunk_words = 0
                for end in range(start + 1, n):
                    chunk_words += sentence_words[end - 1]
                    if chunk_words >= target_words:
                        candidate_ends = [end]
                        break
                if not candidate_ends:
                    candidate_ends = [n]

            for end in candidate_ends:
                words_here = sum(sentence_words[start:end])
                closeness = 1.0 - min(abs(words_here - target_words) / max(target_words, 1), 1.0)
                boundary_strength = strengths[end - 1] if end - 1 < len(strengths) else 1.0
                score = boundary_strength * 1.0 + closeness * 0.35
                if score > best_score:
                    best_score = score
                    best_end = end

            if best_end is None or best_end <= start:
                best_end = min(n, start + 1)

            out.append(" ".join(sentences[start:best_end]).strip())
            start = best_end

        if len(out) >= 2 and word_len(out[-1]) < int(min_words * 0.75):
            out[-2] = f"{out[-2]} {out[-1]}".strip()
            out.pop()

        return [seg for seg in out if str(seg or "").strip()]

    def sliding_semantic_split(
        self,
        segments: List[str],
        *,
        per_chunk_retry: int = 2,
        per_chunk_backoff: float = 0.75
    ) -> List[str]:
        """
        Non-LLM semantic splitting:
        1) sentence-complete split
        2) local lexical-semantic boundary scoring
        3) pack to target chunk_size with relatively uniform lengths
        """
        full_text = " ".join(str(seg or "").strip() for seg in segments if str(seg or "").strip()).strip()
        if not full_text:
            return []
        sentences = self._split_into_semantic_sentences(full_text)
        if not sentences:
            return [full_text]
        return self._semantic_pack_sentences(sentences)

    # ------------------------------------------------------------------ #
    # Pre-split long raw items by max_content_size
    # ------------------------------------------------------------------ #
    def _pre_split_item_by_max_content(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split one raw item into multiple item dicts when content is too long.
        This pre-split runs before prepare_chunk().
        """
        content = item.get("content", "") or ""
        # Keep paragraph/sentence boundaries for natural pre-splitting.
        content = str(content).replace("\r\n", "\n").replace("\r", "\n").strip()

        explicit_id = item.get("_id")
        if explicit_id is None:
            explicit_id = item.get("id")

        if explicit_id is not None and str(explicit_id).strip():
            base_id = str(explicit_id).strip()
        else:
            base_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]

        max_content = max(1, int(self.max_content))
        total_len = word_len(content)

        if not content:
            split_texts = [""]
        elif total_len <= max_content:
            split_texts = [content]
        else:
            # Triggered only when total length exceeds max_content.
            # Split count is auto-decided by max_content upper bound.
            num_segments = max(2, int(math.ceil(total_len / max_content)))
            # Balanced target size; guaranteed <= max_content
            balanced_chunk_size = max(1, int(math.ceil(total_len / num_segments)))

            # Prefer complete paragraphs/sentences before falling back to finer separators.
            natural_separators = [
                "\n\n", "\n",
                "。", "！", "？", "；",
                ". ", "! ", "? ", "; ",
                "，", "、", ", ",
                " ",
                "",
            ]
            pre_splitter = RecursiveCharacterTextSplitter(
                chunk_size=balanced_chunk_size,
                chunk_overlap=0,
                length_function=word_len,
                separators=natural_separators,
            )
            split_texts = pre_splitter.split_text(content) or [content]

            # Hard cap: no segment may exceed max_content.
            hard_cap_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_content,
                chunk_overlap=0,
                length_function=word_len,
                separators=natural_separators,
            )
            capped: List[str] = []
            for seg in split_texts:
                if word_len(seg) <= max_content:
                    capped.append(seg)
                else:
                    capped.extend(hard_cap_splitter.split_text(seg) or [seg])
            split_texts = capped

        out_items: List[Dict[str, Any]] = []
        for idx, seg in enumerate(split_texts, start=1):
            doc_segment_id = f"{base_id}_seg_{idx}"
            new_item = dict(item)
            md = dict(new_item.get("metadata") or {})
            md["raw_doc_id"] = base_id
            md["doc_segment_id"] = doc_segment_id
            new_item["metadata"] = md
            new_item["content"] = seg
            new_item["_id"] = doc_segment_id
            out_items.append(new_item)

        return out_items

    def _generate_segment_title_and_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        For long general documents that were pre-split, turn each segment into a more
        self-contained unit by generating a segment-specific title and metadata.
        """
        original_title = str(item.get("title", "") or "").strip()
        subtitle = str(item.get("subtitle", "") or "").strip()
        text = str(item.get("content", "") or "").strip()
        if not text:
            return item

        try:
            raw = self.document_parser.generate_title_and_metadata(
                text=text,
                title=original_title,
                subtitle=subtitle,
                doc_type=self.doc_type,
            )
            parsed = json.loads(correct_json_format(raw))
        except Exception:
            return item

        generated_title = str(parsed.get("title", "") or "").strip()
        generated_metadata = parsed.get("metadata", {})
        if not isinstance(generated_metadata, dict):
            generated_metadata = {}

        new_item = dict(item)
        new_metadata = dict(new_item.get("metadata") or {})
        if original_title:
            new_metadata.setdefault("original_title", original_title)
        if subtitle:
            new_metadata.setdefault("original_subtitle", subtitle)
        if generated_title and generated_title != original_title:
            new_metadata["generated_segment_title"] = True
        new_metadata.update(generated_metadata)
        new_item["metadata"] = new_metadata
        if generated_title:
            new_item["title"] = generated_title
        return new_item

    def _enrich_pre_split_items(self, _source_item: Dict[str, Any], split_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Only general documents need LLM-generated segment titles after pre-splitting.
        Structured inputs such as screenplay/novel already carry stronger local titles.
        """
        if self.doc_type != "general" or len(split_items) <= 1:
            return split_items

        results_map, _failed = run_concurrent_with_retries(
            items=split_items,
            task_fn=self._generate_segment_title_and_metadata,
            per_task_timeout=120.0,
            max_retry_rounds=1,
            max_in_flight=min(len(split_items), max(1, min(self.max_workers, 8))),
            max_workers=min(len(split_items), max(1, min(self.max_workers, 8))),
            thread_name_prefix="segment_metadata",
            desc_prefix="Generating segment titles",
            show_progress=False,
        )

        enriched: List[Dict[str, Any]] = []
        for idx, item in enumerate(split_items):
            enriched.append(results_map.get(idx, item))
        return enriched

    # ------------------------------------------------------------------ #
    # JSON loader (supports dict or list[dict])
    # ------------------------------------------------------------------ #
    def load_from_json(self, json_file_path: str) -> List[Document]:
        """
        Load input JSON into normalized document dicts.
        Accepts either:
          - list[dict]: each item is one document
          - dict: treated as a single document
        """
        import json as _json
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(
                f"Invalid JSON top-level type: {type(data).__name__}. "
                "Expected a dict or a list of dicts."
            )

        normalized_items: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid item at index {i}: expected dict, got {type(item).__name__}."
                )
            split_items = self._pre_split_item_by_max_content(item)
            normalized_items.extend(self._enrich_pre_split_items(item, split_items))

        documents: List[Document] = []
        for i, item in enumerate(normalized_items):
            documents.append(self._create_document_from_item(item, i))
        return documents

    # ------------------------------------------------------------------ #
    # Core: JSON item -> internal document dict
    # ------------------------------------------------------------------ #
    def _create_document_from_item(self, item: Dict[str, Any], index: int) -> Dict:
        """
        Build a normalized internal document dict from a raw JSON item.
        """
        doc_id = f"doc_{index}"
        title = item.get("title", "") or ""
        subtitle = item.get("subtitle", "") or ""
        raw_text = item.get("content", "") or ""
        version = item.get("version", "") or "default"
        metadata = item.get("metadata", {}) or {}

        explicit_id = item.get("_id")
        if explicit_id is None:
            explicit_id = item.get("id")

        if explicit_id is not None and str(explicit_id).strip():
            raw_source_id = str(explicit_id).strip()
        else:
            content_hash = hashlib.sha1(raw_text.strip().encode("utf-8")).hexdigest()[:16]
            raw_source_id = content_hash

        # Keep a minimal partition marker for downstream compatibility.
        metadata.setdefault("partition", f"{raw_source_id}_1_1")
        raw_doc_id = str(metadata.get("raw_doc_id") or "").strip()
        if not raw_doc_id:
            m = re.match(r"^(.*)_seg_\d+$", raw_source_id)
            raw_doc_id = m.group(1) if m else raw_source_id
        metadata["raw_doc_id"] = raw_doc_id
        metadata.setdefault("doc_segment_id", raw_source_id)

        doc = {
            "id": doc_id,
            "_id": raw_source_id,
            "raw_doc_id": raw_doc_id,
            "doc_type": "document",
            "title": title,
            "version": version,
            "partition": metadata["partition"],
            "subtitle": subtitle,
            "metadata": metadata,
            "content": raw_text.strip()
        }
        return doc

    # ------------------------------------------------------------------ #
    # Convert TextChunk -> Document (for upstream interfaces)
    # ------------------------------------------------------------------ #
    def prepare_document(self, chunk: TextChunk) -> Document:
        """
        Wrap a TextChunk back into a Document-like object for upstream tools.
        """
        return Document(id=chunk.id, content=chunk.content, metadata=chunk.metadata)

    # ------------------------------------------------------------------ #
    # Helpers for timelines
    # ------------------------------------------------------------------ #
    def _parse_timelines_response(self, raw: Any) -> List[str]:
        """
        Parse response of document_parser.extract_time_elements into a list of timeline strings.
        Accepts dict or JSON string. Expected schema: {"timelines": [ ... ]}.
        """
        try:
            data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
        except Exception:
            return []
        tls = data.get("timelines", [])
        if tls is None:
            return []
        if isinstance(tls, list):
            return [x for x in tls if isinstance(x, str) and x.strip()]
        # Fallback: single str
        return [str(tls)] if str(tls).strip() else []

    def _merge_timelines(self, existing: List[str], new_items: List[str]) -> List[str]:
        """
        Stable ordered de-duplication. Preserves first occurrence order across chunks.
        """
        seen = set()
        out: List[str] = []
        for x in chain(existing, new_items):
            k = x.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    def extract_timelines_over_chunks(
        self,
        split_docs: List[str],
        *,
        per_task_timeout: float = 120.0,
        max_retry_rounds: int = 1,
        max_in_flight: Optional[int] = None,
    ) -> List[str]:
        """
        Aggregate timelines by calling document_parser.parse_time_elements
        on each chunk with the current 'existing' list to encourage alignment.
        Returns a single ordered, de-duplicated list at doc level.
        """
        if not split_docs:
            return []

        if max_in_flight is None:
            max_in_flight = int(self.max_workers)

        existing: List[str] = []

        # ---------- Phase A (bootstrap with the first chunk, sequential) ----------
        try:
            existing_str = ", ".join(existing) if existing else ""
            raw0 = self.document_parser.parse_time_elements(split_docs[0], existing_str)
            tls0 = self._parse_timelines_response(raw0)
            existing = self._merge_timelines(existing, tls0)
        except Exception as e:
            # 不吞异常信息，便于定位
            print(f"[WARN] timelines phase A failed: {e}")

        # ---------- Phase B (parallel for the rest, each with a snapshot string) ----------
        def _task(payload):
            idx, text, snapshot_list = payload
            try:
                snapshot_str = ", ".join(snapshot_list) if snapshot_list else ""
                raw = self.document_parser.parse_time_elements(text, snapshot_str)
                return idx, self._parse_timelines_response(raw)
            except Exception as e:
                # 保证单任务失败不影响整体
                print(f"[WARN] timelines phase B task {idx} failed: {e}")
                return idx, []

        items = [(i, split_docs[i], list(existing)) for i in range(1, len(split_docs))]

        results_map, _failed = run_concurrent_with_retries(
            items=items,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=max_retry_rounds,
            max_in_flight=max_in_flight,
            max_workers=max_in_flight,
            thread_name_prefix="timelines",
            desc_prefix="Extracting timelines",
            treat_empty_as_failure=False,
            # 如果你的 run_concurrent_with_retries 没有这个参数，删掉即可
            # show_progress=False
        )

        # Merge results in chunk-index order for stability
        for i in range(1, len(split_docs)):
            if i in results_map:
                _, tls = results_map[i]
                existing = self._merge_timelines(existing, tls)

        return existing


    # ------------------------------------------------------------------ #
    # Unified chunking + sliding-window summary + doc-level metadata + timelines
    # ------------------------------------------------------------------ #
    def prepare_chunk(
        self,
        document: Dict,
        *,
        use_semantic_split: bool = True,
        extract_summary: bool = False,
        extract_metadata: bool = False,
        summary_max_words: int = 200,
    ) -> Dict[str, List[TextChunk]]:
        document_chunks: List[TextChunk] = []

        # ---- Common doc fields ----
        meta = (document.get("metadata") or {}).copy()
        title = document.get("title", "") or ""
        subtitle = document.get("subtitle", "") or ""
        version = document.get("version", "") or "default"
        meta["title"] = title
        meta["subtitle"] = subtitle
        order_val = 0
        if "id" in document:
            try:
                tail = str(document["id"]).split("_")[-1]
                order_val = int(tail)
            except Exception:
                order_val = 0
        meta["order"] = order_val
        meta["version"] = version

        text = document.get("content", "") or ""
        text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

        # ---- (1) Split once ----
        if word_len(text) <= self.chunk_size + 100:
            split_docs = [text]
        else:
            if use_semantic_split:
                split_docs = self.sliding_semantic_split([text])
            else:
                split_docs = self.base_splitter.split_text(text)

        # ---- (2) Document-level summary + metadata ----
        flat_doc_metadata: Dict[str, Any] = {}
        insights_mode = str(self.prepare_insights_mode or "llm").strip().lower()
        if insights_mode == "none":
            insights_mode = "none"
        elif insights_mode not in {"llm", "nlp"}:
            insights_mode = "llm"

        if (extract_summary or extract_metadata) and text and insights_mode == "nlp":
            flat_doc_metadata = self._nlp_metadata(
                text=text,
                title=title or subtitle,
                extract_summary=extract_summary,
                extract_metadata=extract_metadata,
            )
        elif (extract_summary or extract_metadata) and text and insights_mode == "llm":
            try:
                raw_md = self.document_parser.generate_title_and_metadata(
                    text=text,
                    title=title,
                    subtitle=subtitle,
                    doc_type=self.doc_type,
                )
                parsed_md = json.loads(correct_json_format(raw_md))
                md = parsed_md.get("metadata", {})
                if isinstance(md, dict):
                    flat_doc_metadata.update(md)
                generated_title = str(parsed_md.get("title", "") or "").strip()
                if generated_title and generated_title != title:
                    flat_doc_metadata["generated_doc_title"] = generated_title
                if not extract_summary:
                    flat_doc_metadata.pop("summary", None)
                if not extract_metadata:
                    summary_only = flat_doc_metadata.get("summary", None)
                    flat_doc_metadata = {}
                    if summary_only is not None:
                        flat_doc_metadata["summary"] = summary_only
                flat_doc_metadata["prepare_insights_mode"] = "llm"
            except Exception:
                pass
            if extract_metadata:
                flat_doc_metadata = self._inject_model_ner_metadata(
                    flat_doc_metadata,
                    title=title or subtitle,
                    text=text,
                )

        # ---- (3.5) Inject timelines into doc-level metadata (always top-level key)
        # ---- (4) Build chunks ----
        for i, desc in enumerate(split_docs):
            chunk_meta = {
                "chunk_index": i,
                "chunk_type": "document",
                "doc_title": subtitle or title,
                **meta,                # copy from doc-level
                **flat_doc_metadata,   # flatten metadata here (includes timelines if enabled)
            }
            chunk_source_id = (
                str(document.get("_id") or "").strip()
                or str(document.get("id") or "").strip()
            )
            raw_doc_id = str(document.get("raw_doc_id") or meta.get("raw_doc_id") or "").strip()
            if not raw_doc_id and chunk_source_id:
                m = re.match(r"^(.*)_seg_\d+$", chunk_source_id)
                raw_doc_id = m.group(1) if m else chunk_source_id
            if chunk_source_id:
                chunk_meta["doc_segment_id"] = chunk_source_id
            if raw_doc_id:
                chunk_meta["raw_doc_id"] = raw_doc_id

            document_chunks.append(
                TextChunk(
                    id=f"{document['id']}_chunk_{i}",
                    content=desc,
                    source_doc_id=chunk_source_id or document["id"],
                    version=version,
                    metadata=chunk_meta,
                )
            )

        # annotate total chunks
        total = len(document_chunks)
        for ch in document_chunks:
            ch.metadata["total_doc_chunks"] = total

        return {"document_chunks": document_chunks}

    # ------------------------------------------------------------------ #
    # Insights extraction (uses global concurrent executor)
    # ------------------------------------------------------------------ #
    def _parse_insights_response(self, raw: Any) -> List[str]:
        """Parse LLM response into a list of insight strings."""
        try:
            data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
        except Exception:
            return []
        ins = data.get("insights", [])
        if ins is None:
            return []
        if isinstance(ins, list):
            return [x for x in ins if isinstance(x, str)]
        return [str(ins)]

    def _stamp_insight_into_chunk(
        self,
        ch: TextChunk,
        insights: List[str],
        *,
        status: str = "ok",     # "ok" | "timeout" | "exception" | "empty_failed"
        reason: str = "",
        attempt: int = 0,
    ) -> TextChunk:
        """Return a new chunk with insight fields written into metadata."""
        new_meta = dict(ch.metadata or {})
        new_meta["insights"] = insights or []
        new_meta["insights_attempt"] = attempt
        if status == "timeout":
            new_meta["insights_timeout"] = True
        if status != "ok":
            new_meta["insights_failed"] = True
            if reason:
                new_meta["insights_failed_reason"] = reason[:200]
        # Prefer pydantic-like copy API if available
        if hasattr(ch, "model_copy"):
            return ch.model_copy(update={"metadata": new_meta})
        # Fallback for dataclass-like TextChunk
        return ch.copy(update={"metadata": new_meta}, deep=True)

    def extract_insights(
        self,
        chunks: List[TextChunk],
        *,
        per_task_timeout: float = 180.0,
        max_retry_rounds: int = 2,
        max_in_flight: Optional[int] = None,
        treat_empty_as_failure: bool = False,
    ) -> List[TextChunk]:
        """
        Extract insights for a list of TextChunks using the global concurrent executor.
        Returns a list aligned with the input order. Failed/time-out entries are stamped
        with failure flags but still returned (no item is dropped).

        Failure semantics:
          - If `treat_empty_as_failure=True`, empty insights are considered failures
            (status='empty_failed').
        """
        if not chunks:
            return []

        if max_in_flight is None:
            max_in_flight = int(self.max_workers)

        def _task(ch: TextChunk) -> TextChunk:
            # Single-call worker for one chunk
            try:
                raw = self.document_parser.extract_insights(ch.content or "")
                insights = self._parse_insights_response(raw)
                if treat_empty_as_failure and not insights:
                    # mark as empty failure but still return the chunk with flags
                    return self._stamp_insight_into_chunk(
                        ch, [], status="empty_failed", reason="empty insights", attempt=0
                    )
                return self._stamp_insight_into_chunk(ch, insights, status="ok", attempt=0)
            except Exception as e:
                return self._stamp_insight_into_chunk(
                    ch, [], status="exception", reason=str(e), attempt=0
                )

        # Use shared executor; we keep alignment after the run
        results_map, failed_idxs = run_concurrent_with_retries(
            items=chunks,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=max_retry_rounds,
            max_in_flight=max_in_flight,
            max_workers=max_in_flight,
            thread_name_prefix="insights",
            desc_prefix="Extracting insights",
            treat_empty_as_failure=False,  # already handled inside _task via stamping
        )

        # Build aligned output; fill failures gracefully
        out: List[TextChunk] = []
        for i, ch in enumerate(chunks):
            if i in results_map:
                out.append(results_map[i])
            else:
                # If task ultimately failed or timed out across retries, stamp a failure
                out.append(self._stamp_insight_into_chunk(
                    ch, [], status="exception", reason="exhausted_retries", attempt=max_retry_rounds - 1
                ))

        if failed_idxs:
            print(f"[WARN] {len(failed_idxs)} chunks failed after retries: {sorted(failed_idxs)}")

        return out
