# kag/tools/qwen_tools/entity_extraction_tool.py

from typing import Dict, Any, List
import json
from core.utils.format import correct_json_format, is_valid_json
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("extract_entities")
class QwenEntityExtractionTool(BaseTool):
    """实体抽取工具 - Qwen-Agent版本"""
    
    name = "extract_entities"
    description = "从文本中抽取实体（人物、地点、物品、概念等），返回 JSON 格式"
    parameters = [
        {
            "name": "text",
            "type": "string",
            "description": "待抽取实体的原始文本",
            "required": True
        },
        {
            "name": "entity_type_description_text",
            "type": "string",
            "description": "实体类型描述文本，用于指导抽取",
            "required": False
        }
    ]
    
    def __init__(self, prompt_loader=None, llm=None):
        """初始化工具
        
        Args:
            prompt_loader: 提示词加载器
            entity_types_list: 实体类型列表
            llm: 语言模型
        """
        super().__init__()
        self.prompt_loader = prompt_loader
        self.llm = llm
        
    def call(self, params: str, **kwargs) -> str:
        """调用工具
        
        Args:
            params: 工具参数，JSON字符串
            
        Returns:
            抽取结果，JSON字符串
        """
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            abbreviations = params_dict.get("abbreviations", "")
            previous_issues = params_dict("issues", "无")
            previous_suggestions = params_dict("suggestions", "无")

        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})
        
        if not text:
            return json.dumps({"error": "缺少必要参数: text"})
        
        try:

            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_entities_tool_prompt',
                variables={
                    'text': text,
                    'entity_type_description_text': entity_type_description_text,
                    "issues": issues
                },
            )
            
            # 读取agent_prompt作为SystemPrompt
            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={
                    "abbreviations": abbreviations
                }
            )

            suggestion_message = f"在执行知识图谱构建的过程中，以下是一些可供参考的建议：\n{previous_suggestions}\n"

            # 构建消息
            messages = [
                {"role": "system", "content": agent_prompt_text},
                {"role": "assistant", "content": suggestion_mesage}
                {"role": "user", "content": prompt_text}
            ]

            full_response = ""
            max_round = 5
            for i in range(max_round):
                result = self.llm.chat(messages, stream=False)
                content = result[0]['content']
                # print(f"[CHECK] Round {i+1} response: {content}")
                full_response += content.strip()

                # 检查是否是完整 JSON
                if is_valid_json(full_response):
                    return full_response
                # print(f"[CHECK] Round {i+1} response is not valid JSON, trying to correct...")
                # print(f"[CHECK] Round {i+1} response: {content}")

                # 如果不是，加入新的 prompt 请求补全
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "assistant", "content": suggestion_mesage}
                    {"role": "user", "content": "请继续补全上一个 JSON 输出，禁止重复，直接继续输出 JSON 剩余部分："},
                    {"role": "assistant", "content": full_response}
                ]

            repair_prompt = (
                "你之前生成的 JSON 输出不完整，请在不重复已有内容的前提下继续补全它，"
                "确保返回合法、完整、符合 JSON 格式的结构："
            )

            repair_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "assistant", "content": suggestion_mesage}
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": full_response}
            ]

            repair_result = self.llm.chat(repair_messages, stream=False)
            full_response += repair_result[0]['content'].strip()
            # print(f"[CHECK] Repair attempt response: {repaired_text}")

            # 判断是否合法 JSON
            if is_valid_json(full_response):
                return full_response
            else:
                return json.dumps({
                    "error": "补全与修复尝试均失败，仍无法生成合法 JSON。",
                    "partial_result": full_response
                }, ensure_ascii=False)
            # result = self.llm.chat(messages, stream=False)
            # return result[0]['content']
            
        except Exception as e:
            return json.dumps({"error": f"实体抽取失败: {str(e)}"})

