from typing import Optional, List
from kag.utils.config import KAGConfig
from kag.memory.vector_memory import VectorMemory
from ..utils.format import correct_json_format
import json


class DynamicReflector:
    def __init__(self, config: KAGConfig):
        self.config = config.memory
        self.issue_memory = VectorMemory(self.config, "issue_memory")
        self.issue_memory.clear()
        self.suggestion_memory = VectorMemory(self.config, "suggestion_memory")
        self.suggestion_memory.clear()
        
    def generate_logs(self, extraction_result):
        """
        根据抽取结果生成日志
        """
        logs = []

        # 1) 处理实体
        entities = extraction_result.get("entities", [])
        if not entities:
            logs.append("无可识别实体或者实体抽取失败，无相关日志。")
        else:
            for ent in entities:
                name = ent.get("name", "UNKNOWN")
                ent_type = ent.get("type", "UNKNOWN")
                desc = ent.get("description", "") or "空白"
                scope = ent.get("scope", "")
                logs.append(
                    f"实体『{name}』(类型: {ent_type})被抽取出来，相关描述为：{desc}。当前实体scope被标记为：{scope}"
                )

        # 2) 处理关系
        relations = extraction_result.get("relations", [])
        if not relations:
            logs.append("无可识别关系或关系抽取失败，无相关日志。")
        else:
            for rel in relations:
                subj = rel.get("subject", "UNKNOWN")
                obj = rel.get("object", "UNKNOWN")
                relation_type = rel.get("relation_type", "UNKNOWN")
                relation_name = rel.get('relation_name', "UNKNOWN")
                desc = rel.get("description", "") or "空白"
                logs.append(
                    f"实体『{subj}』与实体『{obj}』存在关系「{relation_name}」（类型为{relation_type}），相关描述为：{desc}"
                )

        return logs

    def _store_memory(self, content, reflections):
        issues = reflections.get("current_issues", []) 
        suggestions = reflections.get("suggestions", []) 
        for item in issues:
            _content = f"建议:{item}\n供以下内容参考:\n{content}"
            self.issue_memory.add(text=_content, metadata={"issue": item})
        for item in suggestions:
            _content = f"建议:{item}\n供以下内容参考:\n{content}"
            self.suggestion_memory.add(text=_content, metadata={"suggestion": item})
            
    def _search_relevant_reflections(self, context, k=5):
        related_issues = self.issue_memory.get(context, k)
        # print("[CHECK]: related_issues", related_issues)
        related_issues = [doc.metadata.get("issue") for doc in related_issues]
        related_suggestions = self.suggestion_memory.get(context, k)
        related_suggestions = [doc.metadata.get("suggestion") for doc in related_suggestions]
        return related_issues, related_suggestions
            

