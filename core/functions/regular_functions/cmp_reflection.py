from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class CMPReflector:
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
        self.repair_template = general_repair_template
    
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
            content = params_dict.get("content", "")
            system_prompt = params_dict.get("system_prompt", "")
            previous_reflection = params_dict.get("previous_reflection", {})

            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "feedbacks": [],
                "score": 0
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
     
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="reflect_cmp_extraction_prompt",
                variables={
                    'logs': logs,
                    'content': content,
                },
            )
            messages = [{"role": "system", "content": system_prompt}]
                
                    
            feedbacks = previous_reflection.get("feedbacks", "")
            score = previous_reflection.get("score", "")
            
            if feedbacks and score:
                summary = f"之前反思给出的得分为: {score}\n具体建议为：\n{feedbacks}\n，\n如果现在的抽取结果（后续会给出）有改进，请在此基础上提高分数。"
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
                # logger.info("提取反思完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"提取反思失败",
                    "feedbacks": [],
                    "score": 0
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

            
        except Exception as e:
            logger.error(f"提取反思过程中出现异常: {e}")
            error_result = {
                "error": f"提取反思失败: {str(e)}",
                "feedbacks": [],
                "score": 0
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

