"""
语义分割器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下语义分割结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "segments"字段，且为数组格式
2. 每个分段包含必要的字段信息
3. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


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
        self.repair_template = repair_template
    
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
            max_segment_length = params_dict.get("max_segment_length", 1000)
            overlap_ratio = params_dict.get("overlap_ratio", 0.1)
            split_method = params_dict.get("split_method", "semantic")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "segments": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not text:
            error_result = {"error": "缺少文本内容", "segments": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'max_segment_length': max_segment_length,
                'overlap_ratio': overlap_ratio,
                'split_method': split_method
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('semantic_split_prompt', variables)
            
            # 构建消息
            messages = [{"role": "user", "content": prompt_text}]
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            
            logger.info("语义分割完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"语义分割过程中出现异常: {e}")
            error_result = {
                "error": f"语义分割失败: {str(e)}",
                "segments": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

