# kag/tools/qwen_tools/relation_extraction_tool.py

from typing import Dict, Any, List
import json
from kag.utils.format import correct_json_format, is_valid_json  # ✅ 需确保你有这两个工具函数

from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("extract_relations")
class QwenRelationExtractionTool(BaseTool):
    """关系抽取工具 - Qwen-Agent版本"""

    name = "extract_relations"
    description = "从文本中抽取实体间的关系，返回 JSON 格式"
    parameters = [
        {
            "name": "text",
            "type": "string",
            "description": "待抽取关系的原始文本",
            "required": True
        },
        {
            "name": "entity_list",
            "type": "string",
            "description": "当前文本中已经识别出来的实体",
            "required": False,
        },
        {
            "name": "relation_type_description_text",
            "type": "string",
            "description": "关系类型描述文本，用于指导抽取",
            "required": True
        }
    ]

    def __init__(self, prompt_loader=None, llm=None):
        super().__init__()
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            relation_type_description_text = params_dict.get("relation_type_description_text", "")
            entity_list = params_dict.get("entity_list", "")
            abbreviations = params_dict.get("abbreviations", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not text:
            return json.dumps({"error": "缺少必要参数: text"})

        try:
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_relations_tool_prompt',
                variables={
                    'text': text,
                    'entity_list': entity_list if entity_list else "",
                    'relation_type_description_text': relation_type_description_text,
                },
            )

            # 读取agent_prompt作为SystemPrompt
            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={
                    "abbreviations": abbreviations
                }
            )

            # 构建消息
            messages = [
                {"role": "system", "content": agent_prompt_text},
                {"role": "user", "content": prompt_text}
            ]

            full_response = ""
            max_round = 5
            for i in range(max_round):
                result = self.llm.chat(messages, stream=False)
                content = result[0]['content']
                # print(f"[CHECK] Round {i+1} response: {content}")
                full_response += content.strip()

                if is_valid_json(full_response):
                    return full_response

                # print(f"[CHECK] Round {i+1} response is not valid JSON, trying to continue...")

                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": "请继续补全上一个 JSON 输出，禁止重复，直接继续输出 JSON 剩余部分："},
                    {"role": "assistant", "content": full_response}
                ]

            # 尝试修复输出
            repair_prompt = (
                "你之前生成的 JSON 输出不完整，请在不重复已有内容的前提下继续补全它，"
                "确保返回合法、完整、符合 JSON 格式的结构："
            )

            repair_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": full_response}
            ]

            repair_result = self.llm.chat(repair_messages, stream=False)
            full_response += repair_result[0]['content'].strip()
            
            # print(f"[CHECK] Repair attempt response: {repaired_text}")

            if is_valid_json(full_response):
                return full_response
            else:
                return json.dumps({
                    "error": "补全与修复尝试均失败，仍无法生成合法 JSON。",
                    "partial_result": full_response
                })

        except Exception as e:
            return json.dumps({"error": f"关系抽取失败: {str(e)}"})
