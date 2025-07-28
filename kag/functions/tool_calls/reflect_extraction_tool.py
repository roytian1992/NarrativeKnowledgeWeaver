# kag/tools/qwen_tools/reflect_extraction_tool.py

from typing import Dict, Any, List
import json

from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("reflect_extraction")
class QwenReflectExtractionTool(BaseTool):
    """抽取结果反思工具 - Qwen-Agent版本"""
    
    name = "reflect_extraction"
    description = "对抽取的实体和关系进行质量评估和反思，识别不合适的项目"
    parameters = [
        {
            "name": "entities",
            "type": "string",
            "description": "抽取的实体列表（JSON格式）",
            "required": True
        },
        {
            "name": "relations",
            "type": "string",
            "description": "抽取的关系列表（JSON格式）",
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
            反思结果，JSON字符串
        """
        # 解析参数
        try:
            params_dict = json.loads(params)
            entities = params_dict.get("entities", "")
            relations = params_dict.get("relations", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})
        
        if not entities and not relations:
            return json.dumps({"error": "缺少必要参数: entities 或 relations"})
        
        try:
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='reflect_extraction_tool_prompt',
                variables={
                    'entities': entities,
                    'relations': relations,
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
                required_fields = ["entity_evaluation", "relation_evaluation", "suggestions", "statistics"]
                for field in required_fields:
                    if field not in parsed_result:
                        parsed_result[field] = {}
                
                return json.dumps(parsed_result, ensure_ascii=False)
                
            except json.JSONDecodeError:
                # 如果解析失败，返回原始结果
                return result
            
        except Exception as e:
            return json.dumps({"error": f"抽取反思失败: {str(e)}"})

