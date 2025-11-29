import json
from typing import List, Dict, Any, Optional
from itertools import chain

from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.models.data import Document, TextChunk
from ..utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format, safe_text_for_json
from core.utils.function_manager import run_concurrent_with_retries


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

        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)

        self.chunk_size = config.document_processing.chunk_size
        self.chunk_overlap = config.document_processing.chunk_overlap
        self.max_segments = config.document_processing.max_segments
        self.max_content = config.document_processing.max_content_size
        self.max_workers = config.document_processing.max_workers

        # Base recursive splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
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
            if results and len(tail) < last_min_length:
                # 和前一个块合并
                # 这里简单用拼接，你如果需要可以加换行或空格
                results[-1] = (results[-1] or "") + tail
            else:
                results.append(tail)

        return results

    # ------------------------------------------------------------------ #
    # JSON loader (each input item -> ONE document; no pre-partitions)
    # ------------------------------------------------------------------ #
    def load_from_json(self, json_file_path: str) -> List[Document]:
        """
        Load a JSON list of items into a list of Documents.
        Each input item becomes ONE document (no pre-splitting into partitions).
        """
        import json as _json
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        documents: List[Document] = []
        for i, item in enumerate(data):
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

        # Keep a minimal partition marker for downstream compatibility
        metadata.setdefault("partition", f"{item.get('_id', doc_id)}_1_1")

        doc = {
            "id": doc_id,
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
        extract_timelines: bool = False,          # <--- NEW
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
        meta["order"] = int(str(document["id"]).split("_")[-1]) if "id" in document else 0
        meta["version"] = version

        text = document.get("content", "") or ""
        current_pos = 0

        # ---- (1) Split once ----
        if len(text) <= self.chunk_size + 100:
            split_docs = [text]
        else:
            split_docs = self.base_splitter.split_text(text)
            if use_semantic_split:
                split_docs = self.sliding_semantic_split(split_docs)

        # ---- (1.5) (New) Doc-level timelines from post-split chunks ----
        doc_timelines: List[str] = []
        if extract_timelines and split_docs:
            doc_timelines = self.extract_timelines_over_chunks(split_docs)

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
        if extract_timelines:
            flat_doc_metadata["timelines"] = doc_timelines or []

        # ---- (4) Build chunks ----
        for i, desc in enumerate(split_docs):
            start = current_pos
            end = start + len(desc)

            chunk_meta = {
                "chunk_index": i,
                "chunk_type": "document",
                "doc_title": subtitle or title,
                **meta,                # copy from doc-level
                **flat_doc_metadata,   # flatten metadata here (includes timelines if enabled)
            }

            document_chunks.append(
                TextChunk(
                    id=f"{document['id']}_chunk_{i}",
                    content=desc,
                    document_id=document["id"],
                    version=version,
                    start_pos=start,
                    end_pos=end,
                    metadata=chunk_meta,
                )
            )
            current_pos = end

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
