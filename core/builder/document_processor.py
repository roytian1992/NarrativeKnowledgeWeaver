import json
import hashlib
import math
from typing import List, Dict, Any, Optional, Literal
from itertools import chain

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from core.models.data import Document, TextChunk
from ..utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
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

        # Base recursive splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,  length_function=word_len
        )

        # LLM-powered parser utilities
        self.document_parser = DocumentParser(config, llm)

    # ------------------------------------------------------------------ #
    # Sliding Semantic Split (LLM boundary detection with small retries)
    # ------------------------------------------------------------------ #
    def sliding_semantic_split(
        self,
        segments: List[str],
        *,
        per_chunk_retry: int = 2,
        per_chunk_backoff: float = 0.75
    ) -> List[str]:
        """
        Apply LLM-based boundary detection over merged windows to produce
        discourse-consistent segments, following the paper's "Sliding Semantic Splitting".
        """
        results: List[str] = []
        carry = ""
        last_min_length = 200  # 记录最近一次使用的 min_length（目前恒为 200）

        for i, seg in enumerate(segments):
            # Merge carry with current base chunk to form the candidate window
            text_input = (carry + seg).strip()
            total_len = len(text_input)

            # If too short for a meaningful split, flush old carry and keep seg as new carry
            if total_len < (self.chunk_size + 100) * 0.5:
                if carry:
                    results.append(carry)
                carry = seg
                continue

            max_segments = self.max_segments - 1 if total_len < self.chunk_size + 100 else self.max_segments
            # min_length = max(1, int(total_len / self.max_segments))
            min_length = 200
            last_min_length = min_length  # 记录这次的阈值，后面处理最后一个 segment 用

            payload = {
                "text": safe_text_for_json(text_input),
                "min_length": min_length,
                "max_segments": max_segments
            }

            # Use the global concurrent runner to standardize retry/timeout behavior at call sites.
            # Here, we just implement a local, small retry loop because this is a single-call path.
            last_err: Optional[Exception] = None
            for attempt in range(per_chunk_retry + 1):
                try:
                    raw = self.document_parser.split_text(json.dumps(payload, ensure_ascii=False))
                    parsed = json.loads(correct_json_format(raw))
                    subs = parsed.get("segments", [])
                    if not isinstance(subs, list) or not subs:
                        raise ValueError(f"Splitter returned invalid data on chunk {i}: {subs}")

                    # Append all but the last as finalized segments; keep the last as new carry.
                    results.extend([s for s in subs[:-1] if isinstance(s, str) and s.strip()])
                    carry = subs[-1] if isinstance(subs[-1], str) else ""
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < per_chunk_retry:
                        # small linear backoff
                        import time as _t
                        _t.sleep(min(per_chunk_backoff * (attempt + 1), 2.0))
                    else:
                        # Final failure: surface debug info for diagnosis
                        print("[CHECK] sliding_semantic_split payload:", payload)
                        print("[CHECK] splitter raw result:", locals().get("raw", "N/A"))
                        raise RuntimeError(f"Sliding semantic split failed on chunk {i}: {e}") from e

        # ------ 这里改动：对“最后一个 segment”做 min_length 合并规则 ------
        if isinstance(carry, str) and carry.strip():
            tail = carry.strip()
            if results and word_len(tail) < last_min_length: # 中英文修改
                # 和前一个块合并
                # 这里简单用拼接，你如果需要可以加换行或空格
                results[-1] = (results[-1] or "") + tail
            else:
                results.append(tail)

        return results

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
        text = re.sub(r'\s+', ' ', text).strip()

        # ---- (1) Split once ----
        if len(text) <= self.chunk_size + 100:
            split_docs = [text]
        else:
            split_docs = self.base_splitter.split_text(text)
            if use_semantic_split:
                split_docs = self.sliding_semantic_split(split_docs)

        # ---- (2) Rolling summary -> keep ONLY the final doc-level summary ----
        doc_summary: str = ""
        if extract_summary and split_docs:
            history = ""
            for desc in split_docs:
                try:
                    raw = self.document_parser.summarize_paragraph(desc, summary_max_words, history)
                    parsed = json.loads(correct_json_format(raw))
                    s = parsed.get("summary", "")
                    if not isinstance(s, str):
                        s = str(s) if s is not None else ""
                    history = s
                except Exception:
                    pass
            doc_summary = history
        else:
            doc_summary = ""

        # ---- (3) Document-level metadata from (Title + doc_summary) ----
        flat_doc_metadata: Dict[str, Any] = {}
        if extract_metadata:
            try:
                raw_md = self.document_parser.parse_metadata(
                    doc_summary, title, subtitle, self.doc_type
                )
                parsed_md = json.loads(correct_json_format(raw_md))
                md = parsed_md.get("metadata", {})
                if isinstance(md, dict):
                    flat_doc_metadata.update(md)
                    # put summary at top-level
                    flat_doc_metadata["summary"] = doc_summary
            except Exception:
                pass

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
