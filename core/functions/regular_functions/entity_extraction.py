from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import entity_repair_template, general_rules
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    集成format.py的实体提取器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["entities"]
        self.field_validators = {
            "entities": lambda x: isinstance(x, list) 
        }
        
        # 修复提示词模板
        self.repair_template = entity_repair_template
    
    def call(self, params: str, enable_thinking=True, **kwargs) -> str:
        """
        调用实体提取，保证返回correct_json_format处理后的结果
        
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
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            system_prompt = params_dict.get("system_prompt", "")
            reflection_results = params_dict.get("reflection_results", {})
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "entities": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
        try:
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_entities_prompt',
                variables={
                    'text': text,
                    'entity_type_description_text': entity_type_description_text
                },
            )
           
            # 构造初始消息
            messages = [{"role": "system", "content": system_prompt}]
            # messages.append({"role": "user", "content": f"这是实体和关系抽取时需要遵守的一些准则：\n{general_rules}"})
            
            previous_issues = reflection_results.get("issues", [])
            previous_suggestions = reflection_results.get("suggestions", [])
            previous_results = reflection_results.get("previous_entities", "")
            score = reflection_results.get("score", "")
            
            background_info = f"这是实体和关系抽取时需要遵守的一些准则：\n{general_rules}"
            
            if previous_suggestions:
                previous_suggestions = "\n".join(previous_suggestions)
                background_info +=  f"在执行知识图谱构建的过程中，以下是一些可供参考的建议：\n{previous_suggestions}\n\n"

            if previous_results and previous_issues and score:
                previous_iusses = "\n".join(previous_issues)
                background_info += f"这是你之前抽取的结果，部分内容有待改进： \n{previous_results}, 相关问题为: \n {previous_issues}，得分为: {score}"

            
            messages.append({
                "role": "user",
                "content": background_info
            })

            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                enable_thinking=enable_thinking,
                repair_template=self.repair_template
            )
            if status == "success":
                logger.info("实体提取完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"实体提取失败",
                    "entities": []
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

            
        except Exception as e:
            logger.error(f"实体提取过程中出现异常: {e}")
            error_result = {
                "error": f"实体提取失败: {str(e)}",
                "entities": []
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

