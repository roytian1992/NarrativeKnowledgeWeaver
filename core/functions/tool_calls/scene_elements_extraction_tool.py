# kag/tools/qwen_tools/script_elements_extraction_tool.py

from typing import Dict, Any
import json

from core.utils.format import is_valid_json  # ✅ 确保你有这个工具函数
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("extract_scene_elements")
class QwenSceneElementsExtractionTool(BaseTool):
    """剧本元素抽取工具 - Qwen-Agent版本"""

    name = "extract_scene_elements"
    description = "从剧本文本中抽取特有元素（场景、对话、角色关系等），返回 JSON 格式"
    parameters = [
        {
            "name": "text",
            "type": "string",
            "description": "待抽取元素的剧本文本",
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
            abbreviations = params_dict.get("abbreviations", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not text:
            return json.dumps({"error": "缺少必要参数: text"})

        try:
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_scene_elements_tool_prompt',
                variables={'text': text},
            )

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
                content = result[0]["content"]
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

            # 最后尝试调用 LLM 修复
            repair_prompt = (
                "你上一次生成的 JSON 输出不完整，请在不重复已有内容的前提下继续补全它，"
                "确保返回合法、完整、符合 JSON 格式的结构："
            )

            repair_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": full_response}
            ]

            repair_result = self.llm.chat(repair_messages, stream=False)
            repaired_text = repair_result[0]['content'].strip()
            # print(f"[CHECK] Repair attempt response: {repaired_text}")

            if is_valid_json(repaired_text):
                return repaired_text
            else:
                return json.dumps({
                    "error": "补全与修复尝试均失败，仍无法生成合法 JSON。",
                    "partial_result": full_response,
                    "repair_attempt": repaired_text
                })

        except Exception as e:
            return json.dumps({"error": f"剧本元素抽取失败: {str(e)}"})
