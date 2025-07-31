"""
属性反思器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下属性反思结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "reflection_result"字段，包含反思结果
2. "issues"字段，包含发现的问题列表
3. "suggestions"字段，包含改进建议列表
4. "score"字段，包含评分信息
5. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


class AttributeReflector:
    """
    属性反思器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["reflection_result"]
        self.field_validators = {
            "reflection_result": lambda x: isinstance(x, dict),
            "issues": lambda x: isinstance(x, list) if x is not None else True,
            "suggestions": lambda x: isinstance(x, list) if x is not None else True
        }
        
        # 修复提示词模板
        self.repair_template = repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用属性反思，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数
            params_dict = json.loads(params)
            entity_name = params_dict.get("entity_name", "")
            entity_type = params_dict.get("entity_type", "")
            description = params_dict.get("description", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")
            attributes = params_dict.get("attributes", "")
            abbreviations = params_dict.get("abbreviations", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "reflection_result": {},
                "issues": [],
                "suggestions": [],
                "score": 0
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not entity_name or not attributes:
            error_result = {
                "error": "缺少必要参数", 
                "reflection_result": {},
                "issues": [],
                "suggestions": [],
                "score": 0
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'description': description,
                'attribute_definitions': attribute_definitions,
                'attributes': attributes
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('reflect_attributes_prompt', variables)
            
            # 构建消息
            messages = []
            
            # 添加agent提示
            if abbreviations:
                agent_prompt_text = self.prompt_loader.render_prompt(
                    'agent_prompt',
                    variables={"abbreviations": abbreviations}
                )
                messages.append({"role": "system", "content": agent_prompt_text})
            
            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            
            logger.info("属性反思完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"属性反思过程中出现异常: {e}")
            error_result = {
                "error": f"属性反思失败: {str(e)}",
                "reflection_result": {},
                "issues": [],
                "suggestions": [],
                "score": 0
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

