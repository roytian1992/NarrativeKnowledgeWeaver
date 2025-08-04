from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from kag.utils.general_text import causality_check_repair_template
from kag.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class EventCausalityChecker:
    """
    事件因果关系检查工具：输入两个事件的描述，判断是否存在因果关系，并输出置信度分类结果。
    输出 JSON 格式：
    {
        "causal": "High / Medium / Low / None",
        "reason": "原因或理由的描述"
    }
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["causal", "confidence", "causal", "reverse"]
        self.field_validators = {
            "causal": lambda x: isinstance(x, str) and x in ["High", "Medium", "Low", "None"],
            "reason": lambda x: isinstance(x, str) and len(x.strip()) > 0 if x is not None else True,
            "confidence": lambda x: isinstance(x, float) or isinstance(x, int)
        }
        
        # 修复提示词模板
        self.repair_template = causality_check_repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用因果关系检查，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数
            params_dict = json.loads(params)
            event_1_info = params_dict.get("event_1_info", "")
            event_2_info = params_dict.get("event_2_info", "")
            system_prompt = params_dict.get("system_prompt", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "causal": "None",
                "confidence": 0,
                "reversed": False,
                "reason": f"参数解析失败: {str(e)}"
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not event_1_info or not event_2_info:
            error_result = {
                "error": "缺少必要事件信息", 
                "causal": "None",
                "confidence": 0,
                "reversed": False,
                "reason": "缺少必要的事件信息"
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:

            # 用户提示词（任务具体内容）
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="check_event_causality_prompt",
                variables={
                    "event_1_info": event_1_info,
                    "event_2_info": event_2_info
                }
            )

            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}]
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            if status == "success":
                logger.info("因果关系检查完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"因果关系检查失败",
                    "causal": "None",
                    "confidence": 0,
                    "reversed": False,
                    "reason": ""
                }
                correct_json_format(json.dumps(error_result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"因果关系检查过程中出现异常: {e}")
            # print("[CHECK]")
            error_result = {
                "error": f"因果关系检查失败: {str(e)}",
                "causal": "None",
                "confidence": 0,
                "reversed": False,
                "reason": f"检查过程中出现异常: {str(e)}"
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

