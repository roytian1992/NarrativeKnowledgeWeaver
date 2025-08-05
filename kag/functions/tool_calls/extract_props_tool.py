# kag/tools/qwen_tools/extract_props_tool.py

from typing import Dict, Any, List
import json

from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("extract_props")
class QwenExtractPropsTool(BaseTool):
    """道具抽取工具 - Qwen-Agent版本"""
    
    name = "extract_props"
    description = "从剧本文本中抽取道具信息，包括手持道具、服装道具、场景道具等"
    parameters = [
        {
            "name": "text",
            "type": "string",
            "description": "待抽取道具的剧本文本",
            "required": True
        }
    ]
    
    def __init__(self, prompt_loader=None, llm=None):
        """初始化工具
        
        Args:
            prompt_loader: 提示词加载器
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
        # 解析参数
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})
        
        if not text:
            return json.dumps({"error": "缺少必要参数: text"})
        
        try:
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_props_tool_prompt',
                variables={
                    'text': text,
                },
            )
            
            # 读取agent_prompt作为SystemPrompt
            agent_prompt_data = self.prompt_loader.load_prompt("agent_prompt")
            system_prompt_text = agent_prompt_data["template"]
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": prompt_text}
            ]
            
            # 调用LLM
            result = self.llm.chat(messages)
            
            # 尝试解析返回的JSON，确保格式正确
            try:
                # 提取JSON部分
                if "```json" in result:
                    json_start = result.find("```json") + 7
                    json_end = result.find("```", json_start)
                    json_str = result[json_start:json_end].strip()
                else:
                    json_str = result.strip()
                
                # 验证JSON格式
                parsed_result = json.loads(json_str)
                
                # 确保必要字段存在
                if "props" not in parsed_result:
                    parsed_result["props"] = []
                if "prop_summary" not in parsed_result:
                    parsed_result["prop_summary"] = {
                        "total_count": len(parsed_result["props"]),
                        "by_category": {},
                        "high_priority_count": 0
                    }
                
                return json.dumps(parsed_result, ensure_ascii=False)
                
            except json.JSONDecodeError:
                # 如果解析失败，返回原始结果
                return result
            
        except Exception as e:
            return json.dumps({"error": f"道具抽取失败: {str(e)}"})

