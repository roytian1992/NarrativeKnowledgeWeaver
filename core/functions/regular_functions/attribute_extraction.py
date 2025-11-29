"""
属性提取器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import attribute_repair_template
from core.utils.format import correct_json_format, is_valid_json

logger = logging.getLogger(__name__)

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
            "attributes": lambda x: isinstance(x, dict)
        }
        
        # 修复提示词模板
        self.repair_template = attribute_repair_template
    
    def call(self, params: str,  enable_thinking=True, **kwargs) -> str:
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
            system_prompt = params_dict.get("system_prompt", "")
            feedbacks = params_dict.get("feedbacks", "")
            previous_results = params_dict.get("previous_results", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "attributes": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
                
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_attributes_prompt',
                variables={
                    "text": text,
                    "entity_name": entity_name,
                    "description": description,
                    "entity_type": entity_type,
                    "attribute_definitions": attribute_definitions
                }
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            
            background_info = ""
            if  previous_results and feedbacks:
                background_info += f"上一次抽取的结果如下：\n{previous_results}\n反馈建议如下：\n{feedbacks}\n请仅针对缺失字段或内容错误的字段进行补充，保留已有正确的字段。"
                
                messages.append({
                    "role": "user",
                    "content": background_info
                })
                
                prompt_text = prompt_text + "\n" + f"这是之前抽取的结果：\n {previous_results} \n 在此基础上根据建议进行补充和改进。"


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
            
            # logger.info("属性提取完成，返回格式化后的JSON")
            if status == "success": 
                return corrected_json
            else:
                error_result = {
                    "error": "属性提取失败",
                    "attributes": []
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"属性提取过程中出现异常: {e}")
            error_result = {
                "error": f"属性提取失败: {str(e)}",
                "attributes": []
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

