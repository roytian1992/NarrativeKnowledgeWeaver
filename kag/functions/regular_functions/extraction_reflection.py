from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from kag.utils.general_text import extraction_refletion_repair_template, general_rules
from kag.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class ExtractionReflector:
    """
    提取反思器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = []
        self.field_validators = {}
        
        # 修复提示词模板
        self.repair_template = extraction_refletion_repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用提取反思，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数 - 使用正确的参数名称
            params_dict = json.loads(params)
            logs = params_dict.get("logs", "")
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            relation_type_description_text = params_dict.get("relation_type_description_text", "")
            original_text = params_dict.get("original_text", "")
            system_prompt = params_dict.get("system_prompt", "")
            previous_reflection = params_dict.get("previous_reflection", {})
            version = params_dict.get("version", "default")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "current_issues": [],
                "insights": [],
                "score": 0
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            if version == "short":
                prompt_id = "reflect_extraction_prompt_short"
            else:
                prompt_id = "reflect_extraction_prompt"
                
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id=prompt_id,
                variables={
                    'logs': logs,
                    'entity_type_description_text': entity_type_description_text,
                    'relation_type_description_text': relation_type_description_text
                },
            )
            messages = [{"role": "system", "content": system_prompt}]
            
            
            background_info = ""
            if original_text:
                background_info += f"这是之前信息抽取的原文：\n{original_text.strip()}" 
            background_info += f"这是实体和关系抽取时需要遵守的一些准则：\n{general_rules}"
                
            messages.append({"role": "user", "content": background_info})
                    
            previous_issues = previous_reflection.get("issues", "")
            # previous_suggestions = previous_reflection.get("suggestions", "")
            # relation_extraction_results = previous_reflection.get("previous_relations", "")
            # entitity_extraction_results = previous_reflection.get("previous_entities", "")
            score = previous_reflection.get("score", "")
            
            if previous_issues and score:
                summary = f"之前反思给出的得分为: {score}\n具体建议为：\n{previous_issues}\n，\n如果现在的抽取结果（后续会给出）有改进，请在此基础上提高分数。"
                # summary += f"实体抽取：\n{entitity_extraction_results}\n关系抽取:\n {relation_extraction_results}"
                # print("[CHECK] summary: ", summary)
                messages.append({"role": "user", "content": summary})
            
            
            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=2,
                enable_thinking=True,
                repair_template=self.repair_template
            )
            if status == "success":
                logger.info("提取反思完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"提取反思失败",
                    "current_issues": [],
                    "insights": [],
                    "score": 0
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

            
        except Exception as e:
            logger.error(f"提取反思过程中出现异常: {e}")
            error_result = {
                "error": f"提取反思失败: {str(e)}",
                "current_issues": [],
                "insights": [],
                "score": 0
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

