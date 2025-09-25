from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import general_repair_template, general_rules
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class PropItemExtractor:
    """
    集成format.py的实体提取器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = []
        self.field_validators = {}
        
        # 修复提示词模板
        self.repair_template = general_repair_template
    
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
            content = params_dict.get("content", "")
            system_prompt = params_dict.get("system_prompt", "")
            reflection_results = params_dict.get("reflection_results", {})
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "results": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
        try:
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_propitem_prompt',
                variables={
                    'content': content,
                },
            )
           
            # 构造初始消息
            messages = [{"role": "system", "content": system_prompt}]
            
            feedbacks = reflection_results.get("feedbacks", [])
            previous_results = reflection_results.get("previous_results", "")
            
            background_info = ""
            if previous_results and feedbacks:
                background_info += f"这是你之前抽取的结果，部分内容有待改进： \n{previous_results}, 相关反馈为: \n {feedbacks}"

            
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
                # logger.info("道具提取完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"道具提取失败",
                    "results": []
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

            
        except Exception as e:
            logger.error(f"道具提取过程中出现异常: {e}")
            error_result = {
                "error": f"道具提取失败: {str(e)}",
                "results": []
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

