"""
属性提取器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下属性提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "attributes"字段，且为数组格式
2. 每个属性包含必要的字段信息
3. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


class AttributeExtractor:
    """
    属性提取器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["attributes"]
        self.field_validators = {
            "attributes": lambda x: isinstance(x, list)
        }
        
        # 修复提示词模板
        self.repair_template = repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用属性提取，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            entity_name = params_dict.get("entity_name", "")
            description = params_dict.get("description", "")
            entity_type = params_dict.get("entity_type", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")
            abbreviations = params_dict.get("abbreviations", "")
            feedbacks = params_dict.get("feedbacks", "")
            original_text = params_dict.get("original_text", "")
            previous_results = params_dict.get("previous_results", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "attributes": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not text or not description or not entity_type:
            error_result = {"error": "缺少必要参数", "attributes": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'entity_name': entity_name,
                'description': description,
                'entity_type': entity_type,
                'attribute_definitions': attribute_definitions
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('extract_attributes_prompt', variables)
            
            # 构建消息
            messages = []
            
            # 添加背景信息
            if original_text and previous_results and feedbacks:
                background_info = f"这是提取之前的上下文：\n{original_text}\n这是提取的结果：\n{previous_results}\n相关问题的建议：\n{feedbacks}\n请特别关注上述建议进行属性提取。"
                messages.append({"role": "user", "content": background_info})
            
            # 添加agent提示
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
            
            logger.info("属性提取完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"属性提取过程中出现异常: {e}")
            error_result = {
                "error": f"属性提取失败: {str(e)}",
                "attributes": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

