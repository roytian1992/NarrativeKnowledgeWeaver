from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from kag.utils.general_text import general_repair_template
from kag.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class PlotGenerator:
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
        self.required_fields = ["is_plot", "plot_info", "reason"]
        self.field_validators = {}
        
        # 修复提示词模板
        self.repair_template = general_repair_template
    
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
            event_chain_info = params_dict.get("event_chain_info", "")
            related_context = params_dict.get("related_context", "")
            system_prompt = params_dict.get("system_prompt", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "is_plot": False,
                "reason": "",
                "plot_info": {},
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:

            # 用户提示词（任务具体内容）
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="generate_plot_prompt",
                variables={
                    "event_chain_info": event_chain_info,
                }
            )

            messages = [{"role": "system", "content": system_prompt}]
            if related_context:
                messages.append({"role": "user", "content": f"这是一些可供参考的内容：\n{related_context}"})
                
            messages.append({"role": "user", "content": prompt_text})
            
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
                    "error": f"情节抽取失败",
                    "is_plot": False,
                    "reason": "",
                    "plot_info": {},
                    }
                correct_json_format(json.dumps(error_result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"情节抽取过程中出现异常: {e}")
            # print("[CHECK]")
            error_result = {
                "error": f"情节抽取失败: {str(e)}",
                "is_plot": False,
                "reason": "",
                "plot_info": {},
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

