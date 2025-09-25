"""
语义分割器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

class ParagraphSummarizer:
    """
    语义分割器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["summary"]
        self.field_validators = {
            "summary": lambda x: isinstance(x, str)
        }
        
        # 修复提示词模板
        self.repair_template = general_repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用语义分割，保证返回correct_json_format处理后的结果
        
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
            max_length = params_dict.get("max_length", 200)
            previous_summary = params_dict.get("previous_summary", "")

        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "summary": text}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'max_length': max_length,
            }
            messages = []
            if previous_summary: 
                messages.append({"role": "user", "content": f"这是前文摘要：{previous_summary}"})
            
            # print("[CHECK]读入参数： ", variables)
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('summarize_paragraph_prompt', variables)
            # print("[CHECK] prompt_text: ", prompt_text)
            # 构建消息
            messages.append({"role": "user", "content": prompt_text})
            # print("[CHECK] prompt_text: ", prompt_text)
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=1,
                repair_template=self.repair_template
            )
            
            if status == "success":
                return corrected_json
            else:
                error_result = {
                    "error": f"提取摘要失败，返回整段文本",
                    "summary": text
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"提取摘要过程中出现异常: {e}")
            error_result = {
                "error": f"提取摘要失败: {str(e)}",
                "segments": text
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

