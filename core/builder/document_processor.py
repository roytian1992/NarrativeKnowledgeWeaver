import json
import os
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import deque
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from itertools import chain
from core.models.data import Document, TextChunk
from ..utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format, safe_text_for_json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
import time


class DocumentProcessor:
    """通用文档处理器"""

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self, config: KAGConfig, llm, doc_type="screenplay", max_worker=4):
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

        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # 防止单块文本过长
        self.pre_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_content, chunk_overlap=0)
        self.document_parser = DocumentParser(config, llm)

    # ------------------------------------------------------------------ #
    # 通用：窗口化并发 + 软超时 + 多轮重试
    # ------------------------------------------------------------------ #
    def _run_windowed_with_retries(
        self,
        items: List[Any],
        task_fn: Callable[[Any], Any],
        *,
        per_task_timeout: float = 120.0,
        max_retry_rounds: int = 1,
        max_in_flight: Optional[int] = None,
        desc_prefix: str = "并发任务",
        treat_empty_as_failure: bool = False,
        is_empty_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Tuple[Dict[int, Any], List[int]]:
        """
        通用并发调度器：保持窗口大小、对“真正开始时间”进行软超时控制、失败重试 N 轮。
        返回：({idx: result}, still_failed_indices)
        """
        if max_in_flight is None:
            max_in_flight = int(self.max_workers)

        def _run_round(round_indices: List[int], attempt_idx: int) -> Tuple[Dict[int, Any], List[int]]:
            results_round: Dict[int, Any] = {}
            failures_round: List[int] = []
            todo = deque(round_indices)

            executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"dp_a{attempt_idx}")
            fut_info: Dict[Any, Dict[str, Any]] = {}
            pending = set()

            def _submit_one(idx: int):
                item = items[idx]
                start_box = {"t": None}
                def _wrapper(_item, _box):
                    _box["t"] = time.monotonic()
                    return task_fn(_item)
                f = executor.submit(_wrapper, item, start_box)
                fut_info[f] = {"idx": idx, "start_box": start_box}
                pending.add(f)

            # 初始填满窗口
            while todo and len(pending) < max_in_flight:
                _submit_one(todo.popleft())

            pbar = tqdm(total=len(round_indices), desc=f"{desc_prefix} 第{attempt_idx+1}轮", ncols=100)
            try:
                while pending:
                    done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for f in done:
                        info = fut_info.pop(f)
                        idx = info["idx"]
                        try:
                            res = f.result()
                            # 空结果是否视为失败
                            if treat_empty_as_failure:
                                empty = False
                                if is_empty_fn is not None:
                                    empty = bool(is_empty_fn(res))
                                else:
                                    empty = (res is None) or (res == []) or (res == {})
                                if empty:
                                    failures_round.append(idx)
                                else:
                                    results_round[idx] = res
                            else:
                                results_round[idx] = res
                        except Exception as e:
                            failures_round.append(idx)
                        pending.remove(f)
                        pbar.update(1)
                        # 补位
                        if todo and len(pending) < max_in_flight:
                            _submit_one(todo.popleft())

                    # 软超时判定：只对“已真正开始”计时
                    now = time.monotonic()
                    to_timeout = []
                    for f in pending:
                        sb = fut_info[f]["start_box"]
                        t0 = sb.get("t")
                        if t0 is not None and (now - t0) >= per_task_timeout:
                            to_timeout.append(f)

                    for f in to_timeout:
                        info = fut_info.pop(f)
                        idx = info["idx"]
                        try:
                            f.cancel()
                        finally:
                            failures_round.append(idx)
                            pending.remove(f)
                            pbar.update(1)
                            if todo and len(pending) < max_in_flight:
                                _submit_one(todo.pop() if todo else None)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                pbar.close()

            return results_round, failures_round

        # 第 1 轮
        indices = list(range(len(items)))
        all_results: Dict[int, Any] = {}
        round_results, failures = _run_round(indices, attempt_idx=0)
        all_results.update(round_results)

        # 重试轮
        attempt = 1
        while failures and attempt <= max_retry_rounds:
            round_results, failures = _run_round(failures, attempt_idx=attempt)
            all_results.update(round_results)
            attempt += 1

        # 返回：成功结果 & 未成功的 index 列表
        return all_results, failures

    # ------------------------------------------------------------------ #
    # 语义滑动拆分（增强：小型重试+兜底）
    # ------------------------------------------------------------------ #
    def sliding_semantic_split(
        self,
        segments: List[str],
        *,
        per_chunk_retry: int = 2,
        per_chunk_backoff: float = 0.75
    ) -> List[str]:
        """对初步切段结果进行二次语义切分（可选）"""
        results, carry = [], ""

        for i, seg in enumerate(segments):
            text_input = carry + seg
            total_len = len(text_input)

            # 长度太短：直接输出上一段，当前段暂存
            if total_len < (self.chunk_size + 100) * 0.5:
                if carry:
                    results.append(carry.strip())
                carry = seg
                continue

            max_segments = self.max_segments - 1 if total_len < self.chunk_size + 100 else self.max_segments
            min_length = int(total_len / self.max_segments)

            payload = {
                "text": safe_text_for_json(text_input.strip()),
                "min_length": min_length,
                "max_segments": max_segments
            }

            # 小型重试，避免偶发失败
            last_err = None
            for attempt in range(per_chunk_retry + 1):
                try:
                    result = self.document_parser.split_text(json.dumps(payload, ensure_ascii=False))
                    parsed = json.loads(correct_json_format(result))
                    sub_segments = parsed.get("segments", [])
                    if not isinstance(sub_segments, list) or not sub_segments:
                        raise ValueError(f"splitter returned invalid data on chunk {i}: {sub_segments}")
                    results.extend([s for s in sub_segments[:-1] if isinstance(s, str) and s.strip()])
                    carry = sub_segments[-1]
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < per_chunk_retry:
                        try:
                            time.sleep(min(per_chunk_backoff * (attempt + 1), 2.0))
                        except Exception:
                            pass
                    else:
                        # 最后一次仍失败：抛出详细错误
                        print("[CHECK] payload: ", payload)
                        print("[CHECK] result: ", locals().get("result", "N/A"))
                        raise RuntimeError(f"splitter error on chunk {i}: {e}")

        if isinstance(carry, str) and carry.strip():
            results.append(carry.strip())

        return results

    # ------------------------------------------------------------------ #
    # 元数据抽取（单文档序列版）
    # ------------------------------------------------------------------ #
    def extract_metadata(self, documents: List[Dict]) -> List[Dict]:
        previous_summary = ""
        documents_ = []
        for i, doc in enumerate(documents):
            summary_result = self.document_parser.summarize_paragraph(doc["content"], 200, previous_summary)
            summary_result = json.loads(correct_json_format(summary_result))
            summary = summary_result.get("summary", "")

            metadata_result = self.document_parser.parse_metadata(
                doc["content"], doc.get("title", ""), doc.get("subtitle", ""), self.doc_type
            )
            metadata_result = json.loads(correct_json_format(metadata_result))
            metadata = metadata_result.get("metadata", {})

            previous_summary += summary
            doc["metadata"] = metadata
            documents_.append(doc)
        return documents_

    # ------------------------------------------------------------------ #
    # 元数据抽取（并发+多轮重试版）
    # ------------------------------------------------------------------ #
    def extract_metadata_parallel(
        self,
        document_tracker: Dict[str, List[Dict]],
        *,
        per_task_timeout: float = 300.0,
        max_retry_rounds: int = 1,
        max_in_flight: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        对 document_tracker 中的每个文档列表并发执行 extract_metadata，窗口化调度，失败可重试。
        返回同结构的 result_tracker；失败的键返回 []。
        """
        kv_pairs = list(document_tracker.items())

        def _task(pair):
            key, doc_list = pair
            return self.extract_metadata(doc_list)

        # treat_empty_as_failure: 抽取空也算失败（可按需调整）
        all_results_idx, still_failed_idx = self._run_windowed_with_retries(
            items=kv_pairs,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=max_retry_rounds,
            max_in_flight=max_in_flight or self.max_workers,
            desc_prefix="元数据抽取中",
            treat_empty_as_failure=False,  # 抽空不一定是错误，保留为 False
        )

        result_tracker: Dict[str, List[Dict]] = {}
        for i, (key, _) in enumerate(kv_pairs):
            if i in all_results_idx:
                result_tracker[key] = all_results_idx[i]
            else:
                print(f"[!] 抽取 {key} 失败（重试后仍失败），返回空列表")
                result_tracker[key] = []

        return result_tracker

    # ------------------------------------------------------------------ #
    # 洞见抽取
    # ------------------------------------------------------------------ #
    def extract_insights(
        self,
        chunks: List[TextChunk],
        *,
        per_task_timeout: float = 120.0,
        max_retry_rounds: int = 1,
        treat_empty_as_failure: bool = False,
        max_in_flight: int = None,   # 并发窗口；默认用 self.max_workers
    ) -> List[TextChunk]:

        if max_in_flight is None:
            max_in_flight = int(self.max_workers)

        def _stamp_result(
            ch: TextChunk,
            insights: List[str],
            attempt: int,
            status: str = "ok",   # ok | timeout | exception | empty_failed
            reason: str = "",
        ) -> TextChunk:
            new_meta = dict(ch.metadata or {})
            new_meta["insights"] = insights or []
            new_meta["insights_attempt"] = attempt
            if status == "timeout":
                new_meta["insights_timeout"] = True
            if status != "ok":
                new_meta["insights_failed"] = True
                if reason:
                    new_meta["insights_failed_reason"] = reason
            if hasattr(ch, "model_copy"):
                return ch.model_copy(update={"metadata": new_meta})
            else:
                return ch.copy(update={"metadata": new_meta}, deep=True)

        def _parse_insights(raw: Any) -> List[str]:
            data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
            ins = data.get("insights", [])
            if ins is None:
                return []
            if isinstance(ins, list):
                return [x for x in ins if isinstance(x, str)]
            return [str(ins)]

        def _run_wrapper(ch: TextChunk, start_box: Dict[str, float]) -> List[str]:
            start_box["t"] = time.monotonic()
            raw = self.document_parser.extract_insights(ch.content or "")
            return _parse_insights(raw)

        def _run_round(round_chunks: List[TextChunk], attempt_idx: int) -> Tuple[List[TextChunk], List[TextChunk]]:
            results_in_order: List[TextChunk] = []
            failures: List[TextChunk] = []
            todo = deque(round_chunks)

            executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"insight_a{attempt_idx}")
            fut_info: Dict[Any, Dict[str, Any]] = {}
            pending = set()

            def _submit_one(ch: TextChunk):
                start_box = {"t": None}
                f = executor.submit(_run_wrapper, ch, start_box)
                fut_info[f] = {
                    "ch": ch,
                    "attempt": attempt_idx,
                    "start_box": start_box,
                }
                pending.add(f)

            while todo and len(pending) < max_in_flight:
                _submit_one(todo.popleft())

            pbar = tqdm(total=len(round_chunks), desc=f"洞见抽取 第{attempt_idx+1}轮", ncols=100)

            try:
                while pending:
                    done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for f in done:
                        info = fut_info.pop(f)
                        ch = info["ch"]
                        try:
                            insights = f.result()
                            if treat_empty_as_failure and not insights:
                                failures.append(ch)
                                results_in_order.append(_stamp_result(
                                    ch, [], attempt=attempt_idx, status="empty_failed", reason="empty insights"
                                ))
                            else:
                                results_in_order.append(_stamp_result(
                                    ch, insights, attempt=attempt_idx, status="ok"
                                ))
                        except Exception as e:
                            failures.append(ch)
                            results_in_order.append(_stamp_result(
                                ch, [], attempt=attempt_idx, status="exception", reason=str(e)[:200]
                            ))
                        pending.remove(f)
                        pbar.update(1)
                        if todo and len(pending) < max_in_flight:
                            _submit_one(todo.popleft())

                    now = time.monotonic()
                    to_timeout = []
                    for f in pending:
                        sb = fut_info[f]["start_box"]
                        t0 = sb.get("t")
                        if t0 is not None and (now - t0) >= per_task_timeout:
                            to_timeout.append(f)

                    for f in to_timeout:
                        info = fut_info.pop(f)
                        ch = info["ch"]
                        try:
                            f.cancel()
                        finally:
                            failures.append(ch)
                            results_in_order.append(_stamp_result(
                                ch, [], attempt=attempt_idx, status="timeout",
                                reason=f"soft_timeout_{per_task_timeout}s"
                            ))
                            pending.remove(f)
                            pbar.update(1)
                            if todo and len(pending) < max_in_flight:
                                _submit_one(todo.popleft())
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                pbar.close()

            return results_in_order, failures

        all_results: List[TextChunk] = []
        round_results, failures = _run_round(chunks, attempt_idx=0)
        all_results.extend(round_results)

        attempt = 1
        while failures and attempt <= max_retry_rounds:
            round_results, failures = _run_round(failures, attempt_idx=attempt)
            all_results.extend(round_results)
            attempt += 1

        if failures:
            for ch in failures:
                all_results.append(_stamp_result(
                    ch, [], attempt=attempt-1, status="exception", reason="exhausted_retries"
                ))

        return all_results

    # ------------------------------------------------------------------ #
    # 加载 JSON 文件
    # ------------------------------------------------------------------ #
    def extract_metadata_wrapper(self, documents: List[Dict]) -> List[Dict]:
        """可单独暴露的包装（保留原函数名以兼容）"""
        return self.extract_metadata(documents)

    def load_from_json(self, json_file_path: str, extract_metadata: bool = False) -> List[Document]:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.document_tracker = dict()
        for data_ in data:
            content = data_.get("content", "")
            if len(content) >= self.max_content:
                text_chunks = self.pre_splitter.split_text(content)
                if len(text_chunks[-1]) <= 0.25 * self.max_content and len(text_chunks) >= 2:
                    text_chunks[-2] += text_chunks[-1]
                    text_chunks.pop()
            else:
                text_chunks = [content]

            num_partitions = len(text_chunks)
            for i, text_chunk in enumerate(text_chunks):
                copy_ = data_.copy()
                copy_["content"] = text_chunk
                copy_["metadata"] = copy_.get("metadata", {})
                copy_["metadata"]["partition"] = f"{data_['_id']}_{i+1}_{num_partitions}"

                if data_["_id"] not in self.document_tracker:
                    self.document_tracker[data_["_id"]] = [copy_]
                else:
                    self.document_tracker[data_["_id"]].append(copy_)

        if extract_metadata:
            self.document_tracker = self.extract_metadata_parallel(
                self.document_tracker,
                per_task_timeout=300.0,
                max_retry_rounds=1,
                max_in_flight=self.max_workers
            )

        data_chunks = list(chain.from_iterable(self.document_tracker.values()))

        documents: List[Document] = []
        for i, item in enumerate(data_chunks):
            doc_dict = self._create_document_from_item(item, i)
            documents.append(doc_dict)

        return documents

    # ------------------------------------------------------------------ #
    # 核心：JSON -> 内部文档 dict
    # ------------------------------------------------------------------ #
    def _create_document_from_item(self, item: Dict[str, Any], index: int) -> Dict:
        doc_id = f"doc_{index}"
        title = item.get("title", "")
        subtitle = item.get("subtitle", "")
        raw_text = item.get("content", "")
        metadata = item.get("metadata", {}) or {}
        partition = metadata.get("partition", f"{doc_id}-1")

        doc = {
            "id": doc_id,
            "doc_type": "document",
            "title": title,
            "partition": partition,
            "subtitle": subtitle,
            "metadata": metadata,
            "content": raw_text.strip()
        }

        return doc

    # ------------------------------------------------------------------ #
    # TextChunk -> Document（上游接口所需）
    # ------------------------------------------------------------------ #
    def prepare_document(self, chunk: TextChunk) -> Document:
        return Document(id=chunk.id, content=chunk.content, metadata=chunk.metadata)

    # ------------------------------------------------------------------ #
    # 分块策略
    # ------------------------------------------------------------------ #
    def prepare_chunk(self, document: Dict) -> Dict[str, List[TextChunk]]:
        document_chunks: List[TextChunk] = []

        meta = document.get("metadata", {}).copy()
        title = document.get("title", "")
        subtitle = document.get("subtitle", "")

        meta["title"] = title
        meta["subtitle"] = subtitle
        meta["order"] = int(document['id'].split("_")[-1])

        document_content = document.get("content", "")
        chunk_index, current_pos = 0, 0

        # --- 描述块处理 ---
        if len(document_content) <= self.chunk_size + 100:
            split_docs = [document_content]
        else:
            split_docs = self.base_splitter.split_text(document_content)
            split_docs = self.sliding_semantic_split(split_docs)

        for desc in split_docs:
            start = document_content.find(desc, current_pos)
            end = start + len(desc)
            document_chunks.append(
                TextChunk(
                    id=f"{document['id']}_chunk_{chunk_index}",
                    content=desc,
                    document_id=document["id"],
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        "chunk_index": chunk_index,
                        "chunk_type": "document",
                        "doc_title": subtitle or title,
                        **meta,
                    },
                )
            )
            chunk_index += 1
            current_pos = end

        for chunk in document_chunks:
            chunk.metadata["total_doc_chunks"] = len(document_chunks)

        return {
            "document_chunks": document_chunks
        }
