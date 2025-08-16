import json
import os
from typing import List, Dict, Any, Optional

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
        
        self.pre_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_content, chunk_overlap=0) # 防止单块文本过长。
        self.document_parser = DocumentParser(config, llm)
        
    # ------------------------------------------------------------------ #
    # 语义滑动拆分
    # ------------------------------------------------------------------ #
    def sliding_semantic_split(self, segments: List[str]) -> List[str]:
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

            payload = {"text": safe_text_for_json(text_input.strip()), "min_length": min_length, "max_segments": max_segments} 
            
            try:
                result = self.document_parser.split_text(json.dumps(payload, ensure_ascii=False))
                parsed = json.loads(correct_json_format(result))
                sub_segments = parsed.get("segments", [])
            except Exception as e:
                print("[CHECK] payload: ", payload)
                print("[CHECK] result: ", result)
                raise RuntimeError(f"splitter error on chunk {i}: {e}")

            if not isinstance(sub_segments, list) or not sub_segments:
                raise ValueError(f"splitter returned invalid data on chunk {i}: {sub_segments}")

            results.extend(sub_segments[:-1])
            carry = sub_segments[-1]

        if carry.strip():
            results.append(carry.strip())

        return results

    # ------------------------------------------------------------------ #
    # 加载 JSON 文件
    # ------------------------------------------------------------------ #
    def extract_metadata(self, documents: List[Dict]):
        previous_summary = ""
        documents_ = []
        for i, doc in enumerate(documents):
            summary_result = self.document_parser.summarize_paragraph(doc["content"], 200, previous_summary)
            summary_result = json.loads(correct_json_format(summary_result))
            summary = summary_result.get("summary", "")
            metadata_result = self.document_parser.parse_metadata(doc["content"], doc.get("title", ""), doc.get("subtitle", ""), self.doc_type)
            metadata_result = json.loads(correct_json_format(metadata_result))
            metadata = metadata_result.get("metadata", {})
            previous_summary += summary
            doc["metadata"] = metadata
            documents_.append(doc)
        return documents_
    
    def extract_metadata_parallel(self, document_tracker: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        对 document_tracker 中的每个文档列表并发执行 extract_metadata，并显示进度条
        """
        result_tracker: Dict[str, List[Dict]] = {}
        total_tasks = len(document_tracker)

        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = {
                executor.submit(self.extract_metadata, doc_list): key
                for key, doc_list in document_tracker.items()
            }

            for future in tqdm(as_completed(futures), total=total_tasks, desc="元数据抽取中", ncols=100):
                key = futures[future]
                try:
                    result = future.result()
                    result_tracker[key] = result
                except Exception as e:
                    print(f"[!] 抽取 {key} 失败: {e}")
                    result_tracker[key] = []

        return result_tracker
    

    def extract_insights(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        并发为每个 TextChunk 生成 metadata['insights']，
        以【完成顺序】返回；单任务软超时=120s（可改），超时则降级占位并继续。
        """

        PER_TASK_TIMEOUT = 120.0  # 秒；你可以保持 180

        def _make_with_insights(ch: TextChunk, insights, timeout=False) -> TextChunk:
            new_meta = dict(ch.metadata or {})
            new_meta["insights"] = insights or []
            if timeout:
                new_meta["insights_timeout"] = True
            if hasattr(ch, "model_copy"):  # pydantic v2
                return ch.model_copy(update={"metadata": new_meta})
            else:  # pydantic v1
                return ch.copy(update={"metadata": new_meta}, deep=True)

        def _run(ch: TextChunk) -> TextChunk:
            insights = []
            try:
                raw = self.document_parser.extract_insights(ch.content or "")
                data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
                ins = data.get("insights", [])
                if ins is None:
                    insights = []
                elif isinstance(ins, list):
                    insights = [x for x in ins if isinstance(x, str)]
                else:
                    insights = [str(ins)]
            except Exception:
                insights = []
            return _make_with_insights(ch, insights, timeout=False)

        results: List[TextChunk] = []

        # 不用 with，上下文会等待所有任务完成；我们要能不等待地脱身
        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="insight")
        try:
            fut_info = {}  # future -> {"start": float, "ch": TextChunk}
            for ch in chunks:
                f = executor.submit(_run, ch)
                fut_info[f] = {"start": time.time(), "ch": ch}

            pbar = tqdm(total=len(fut_info), desc="并发抽取洞见中", ncols=100)

            pending = set(fut_info.keys())
            # 循环直到我们“收集”够所有结果（包括超时占位）
            while pending:
                # 1) 先收集已经完成的
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        res = f.result()  # 这里不会阻塞，已完成
                    except Exception:
                        # 失败也降级
                        ch = fut_info[f]["ch"]
                        res = _make_with_insights(ch, [], timeout=True)
                    results.append(res)
                    pbar.update(1)
                    fut_info.pop(f, None)

                # 2) 再检查还在跑的：是否超过软超时？超过就降级并不再等待
                now = time.time()
                to_forget = []
                for f in pending:
                    start = fut_info[f]["start"]
                    if now - start >= PER_TASK_TIMEOUT:
                        ch = fut_info[f]["ch"]
                        # 尝试取消（若已在运行会返回 False），但我们不再等待它
                        f.cancel()
                        res = _make_with_insights(ch, [], timeout=True)
                        results.append(res)
                        pbar.update(1)
                        to_forget.append(f)
                if to_forget:
                    for f in to_forget:
                        pending.remove(f)
                        fut_info.pop(f, None)

            pbar.close()
        finally:
            # 不等待未完成线程，且取消队列里尚未开始的任务，避免退出卡住
            executor.shutdown(wait=False, cancel_futures=True)

        return results
    
    
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
                # copy_["partition"] = f"{data_["_id"]}-{i+1}-{num_partitions}"
                copy_["metadata"] = copy_.get("metadata", {})
                copy_["metadata"]["partition"] = f"{data_['_id']}_{i+1}_{num_partitions}"

                if data_["_id"] not in self.document_tracker:
                    self.document_tracker[data_["_id"]] = [copy_]
                else:
                    self.document_tracker[data_["_id"]].append(copy_)

        if extract_metadata:
            self.document_tracker = self.extract_metadata_parallel(self.document_tracker)
        
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
        # partition = item.get("partition", f"{doc_id}-1")
        metadata = item.get("metadata", {}) or {}
        partition = metadata.get("partition", f"{doc_id}-1")

        conversations = item.get("conversations", []) or []

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
        # print("[CHECK] doc id: ", document["id"])
        document_chunks: List[TextChunk] = []

        meta = document.get("metadata", {}).copy()
        title = document.get("title", "")
        subtitle = document.get("subtitle", "")
        partition = document.get("partition", "")
        # document_id = meta.get("id", "")

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
                        # "partition": partition,
                        **meta,
                    },
                )
            )
            chunk_index += 1
            current_pos = end

        # --- 写入汇总信息 ---
        for chunk in document_chunks:
            chunk.metadata["total_doc_chunks"] = len(document_chunks)

        return {
            "document_chunks": document_chunks
        }
