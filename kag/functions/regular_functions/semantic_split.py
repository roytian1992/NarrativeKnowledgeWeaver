"""
语义分割器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from kag.utils.general_text import semantic_splitter_repair_template

logger = logging.getLogger(__name__)

class SemanticSplitter:
    """
    语义分割器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["segments"]
        self.field_validators = {
            "segments": lambda x: isinstance(x, list) and len(x) > 0
        }
        
        # 修复提示词模板
        self.repair_template = semantic_splitter_repair_template
    
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
            max_segments = params_dict.get("max_segments", 3)
            min_length = params_dict.get("min_length", len(text) * 0.4)
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "segments": [text, ]}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'max_segments': max_segments,
                'min_length': min_length,
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('semantic_splitter_prompt', variables)
            
            # 构建消息
            messages = [{"role": "user", "content": prompt_text}]
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=1,
                repair_template=self.repair_template
            )
            
            logger.info("语义分割完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"语义分割过程中出现异常: {e}")
            error_result = {
                "error": f"语义分割失败: {str(e)}",
                "segments": [text, ]
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

