"""
属性反思器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import attribute_reflection_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class AttributeReflector:
    """
    属性反思器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["feedbacks", "score"]
        self.field_validators = {}
        
        # 修复提示词模板
        self.repair_template = attribute_reflection_repair_template
    
    def call(self, params: str, enable_thinking=True, **kwargs) -> str:
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
            attributes = params_dict.get("attributes", "")
            description = params_dict.get("description", "")
            entity_type = params_dict.get("entity_type", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")
            system_prompt = params_dict.get("system_prompt", "")  # 和实体抽取逻辑保持一致
            original_text = params_dict.get("original_text", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "feedbacks": [],
                "score": 0,
                "attributes_to_retry": [],
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                

        try:

            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='reflect_attributes_prompt',
                variables={
                    "attributes": attributes,
                    "description": description,
                    "entity_type": entity_type,
                    "attribute_definitions": attribute_definitions
                }
            )
            
            # agent 指令（system prompt），同你之前写法
            messages = [{"role": "system", "content": system_prompt}]
            
            if original_text:
                 messages.append({
                    "role": "user",
                    "content": f"这些是原始文本：\n{original_text}\n\n请在此基础上进行属性抽取。"
                })

            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template,
                enable_thinking=enable_thinking
            )
            
            # logger.info("属性反思完成，返回格式化后的JSON")
            if status != "error":
                return corrected_json
            else:
                error_result = {
                    "error": f"属性反思失败: {str(e)}",
                    "feedbacks": [],
                    "score": 0,
                    "attributes_to_retry": [],
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
            
        except Exception as e:
            logger.error(f"属性反思过程中出现异常: {e}")
            error_result = {
                "error": f"属性反思失败: {str(e)}",
                "feedbacks": [],
                "score": 0,
                "attributes_to_retry": [],
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

