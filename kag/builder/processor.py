"""
文档处理器模块

支持剧本格式的JSON数据处理
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import os
from ..models.entities import Document, TextChunk
from ..models.script_models import ScriptDocument, ScriptContentParser, SceneMetadata
from ..utils.config import KAGConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from kag.utils.prompt_loader import PromptLoader
from kag.functions.regular_functions import SemanticSplitter


class DocumentProcessor:
    """文档处理器，支持剧本格式"""
    
    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
        self.prompt_loader = PromptLoader(prompt_dir)
        
        self.chunk_size = config.extraction.chunk_size
        self.chunk_overlap = config.extraction.chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.advanced_splitter = SemanticSplitter(self.prompt_loader, llm)
        self.max_segments = 3
        
    def sliding_semantic_split(self, segments: List[str]) -> List[str]:
        """
        基于已有初始 segments 列表进行语义滑动拆分，每次拼接前后段后调用 splitter.call()。
        
        参数：
            segments: 初步切分的段落列表（如等长切段）
        返回：
            一个语义一致的子段列表
        """
        results = []
        carry = ""

        for i, seg in enumerate(segments):
            # 拼接上一次遗留的内容
            text_input = carry + seg
            total_len = len(text_input)

            # print(f"[CHECK]拼接后的长度: {total_len}")

            # 新增优化：若拼接长度过短，直接返回原段落
            if total_len < (self.chunk_size + 100) * 0.5:
                if carry:
                    results.append(carry.strip())
                carry = seg  # 当前段保留为下一轮拼接（注意不是 text_input）
                continue

            # 动态控制最多拆分段数
            if total_len < self.chunk_size + 100:
                max_segments = self.max_segments - 1
            else:
                max_segments = self.max_segments

            # 设置最小长度（防止切得过短）
            min_length = total_len * (1 / self.max_segments)

            payload = {
                "text": text_input,
                "min_length": int(min_length),
                "max_segments": max_segments
            }

            try:
                response = self.advanced_splitter.call(json.dumps(payload))
                parsed = json.loads(response)
                sub_segments = parsed.get("segments", [])
                # print("[CHECK] sub_segments: ", sub_segments)
            except Exception as e:
                raise RuntimeError(f"第 {i} 段调用 splitter 异常: {e}")

            if not sub_segments or not isinstance(sub_segments, list):
                print("[CHECK] response: ", response)
                # print("[CHECK] sub_segments: ", sub_segments)
                raise ValueError(f"第 {i} 段拆分结果格式错误: {sub_segments}")

            # 取前 n-1 段为有效输出，最后一段为下轮拼接保留
            results.extend(sub_segments[:-1])
            carry = sub_segments[-1]

        # 如果最后一段仍有内容，补充进去
        if carry.strip():
            results.append(carry.strip())

        return results


        
    def load_from_json(self, json_file_path: str) -> List[Document]:
        """从JSON文件加载文档，支持剧本格式"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        for item in data:
            doc = self._create_document_from_item(item, len(documents))
            if doc:
                documents.append(doc)
                    
        return documents

    
    def _is_single_document(self, data: Dict[str, Any]) -> bool:
        """判断是否为单个文档"""
        return any(key in data for key in ['id', '_id', 'content', 'scene_name'])
    
    def _create_document_from_item(self, item: Dict[str, Any], index: int) -> Optional[Document]:
        """从数据项创建文档"""
        # 处理ID字段（支持_id和id）
        doc_id = str(item.get('_id', item.get('id', f'doc_{index}')))
        return self._create_script_document(item, doc_id)
    
    def _is_script_format(self, item: Dict[str, Any]) -> bool:
        """判断是否为剧本格式"""
        script_fields = ['scene_name', 'conversation', 'sub_scene_name']
        return any(field in item for field in script_fields)
    
    def _create_script_document(self, item: Dict[str, Any], doc_id: str) -> Dict:
        """创建剧本文档"""
        # 使用ScriptDocument解析
        script_doc = ScriptDocument.from_script_data(item)
        
        # 解析剧本内容
        content_sections = ScriptContentParser.parse_content_sections(script_doc.content)
        
        raw_content = self._build_content(script_doc, content_sections)
        
        # 创建文档数据
        content = {
            'doc_type': 'script',
            'scene_name': script_doc.scene_name,
            'sub_scene_name': script_doc.sub_scene_name,
            'conversations': [conv.dict() for conv in script_doc.conversations],
            'content_sections': content_sections
        }
        
        # 添加场景元数据
        if script_doc.scene_metadata:
            content['scene_metadata'] = script_doc.scene_metadata.dict()
            # content.update({
            #     'scene_number': script_doc.scene_metadata.scene_number,
            #     'scene_type': script_doc.scene_metadata.scene_type,
            #     'time_of_day': script_doc.scene_metadata.time_of_day,
            #     'location': script_doc.scene_metadata.location,
            #     'sub_location': script_doc.scene_metadata.sub_location
            # })
        
        # 添加角色信息
        if script_doc.conversations:
            characters = list(set([conv.character for conv in script_doc.conversations]))
            content['characters'] = characters
            content['dialogue_count'] = len(script_doc.conversations)
        
        return dict(
            id=doc_id,
            raw_content=raw_content,
            content=content
        )
    
    def _build_content(self, script_doc: ScriptDocument, content_sections: Dict[str, List[str]]) -> str:
        """构建增强的剧本内容"""
        parts = []
        parts.append(script_doc.content)
        return "\n".join(parts)
    
    def prepare_document(self, chunk: TextChunk) -> Document:
        return Document(id=chunk.id, content=chunk.content, metadata=chunk.metadata)
    
    def prepare_chunk(self, document: dict) -> Dict[str, List[TextChunk]]:
        """剧本文档的分块策略：返回按类型组织的 chunk 字典（已移除舞台指示）"""
        description_chunks = []
        conversation_chunks = []

        scene_metadata = document["content"].get('scene_metadata', {})
        conversations = document["content"].get('conversations', [])
        # content_sections = document["content"].get('content_sections', {})
        scene_name = document["content"].get('scene_name', "")
        sub_scene_name = document["content"].get('sub_scene_name', "")
        scene_metadata["scene_name"] = scene_name
        scene_metadata["sub_scene_name"] = sub_scene_name

        descriptions = document.get("raw_content", "")

        chunk_index = 0
        current_pos = 0

        # 分割场景描述
        if len(descriptions) <= self.chunk_size + 100:
            split_docs = [descriptions]
        else:
            split_docs = self.base_splitter.split_text(descriptions)
            split_docs = self.sliding_semantic_split(split_docs)

        for desc_content in split_docs:
            start = descriptions.find(desc_content, current_pos)
            end = start + len(desc_content)
            chunk = TextChunk(
                id=f"{document['id']}_chunk_{chunk_index}",
                content=f"{desc_content}",
                document_id=document["id"],
                start_pos=start,
                end_pos=end,
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_type": "scene_description",
                    "play_name": sub_scene_name or scene_name,
                    **scene_metadata
                }
            )
            description_chunks.append(chunk)
            chunk_index += 1
            current_pos = end

        # 分割对话块
        for c in conversations:
            content = f"{c['character']}：{c['content']}"
            if c.get('dialogue_type'):
                content += f" ({c['dialogue_type']})"
            if c.get('remarks'):
                content += f" [{'，'.join(c['remarks'])}]"

            chunk = TextChunk(
                id=f"{document['id']}_chunk_{chunk_index}",
                content=content,
                document_id=document["id"],
                start_pos=0,
                end_pos=len(c['content']),
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_type": "dialogue",
                    "character": c['character'],
                    "type": c.get('dialogue_type'),
                    "remark": c.get('remarks'),
                    **scene_metadata
                }
            )
            conversation_chunks.append(chunk)
            chunk_index += 1

        # ✅ 添加分类型总数 metadata
        total_description_chunks = len(description_chunks)
        total_conversation_chunks = len(conversation_chunks)

        for chunk in description_chunks:
            chunk.metadata["total_description_chunks"] = total_description_chunks
        for chunk in conversation_chunks:
            chunk.metadata["total_conversation_chunks"] = total_conversation_chunks

        return {
            "description_chunks": description_chunks,
            "conversation_chunks": conversation_chunks
        }
