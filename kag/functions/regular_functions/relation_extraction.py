"""
关系提取器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下关系提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "relations"字段，且为数组格式
2. 每个关系包含必要的字段信息
3. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


class RelationExtractor:
    """
    关系提取器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["relations"]
        self.field_validators = {
            "relations": lambda x: isinstance(x, list)
        }
        
        # 修复提示词模板
        self.repair_template = repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用关系提取，保证返回correct_json_format处理后的结果
        
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
            entity_list = params_dict.get("entity_list", "")
            abbreviations = params_dict.get("abbreviations", "")
            reflection_results = params_dict.get("reflection_results", {})
            entity_extraction_results = params_dict.get("entity_extraction_results", {})
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "relations": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not text:
            error_result = {"error": "缺少文本内容", "relations": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'entity_list': entity_list,
                'relation_type_description_text': "",  # 可以从参数中获取
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('extract_relations_prompt', variables)
            
            # 构建消息
            messages = []
            
            # 添加反思信息作为背景
            previous_issues = reflection_results.get("issues", "")
            previous_suggestions = reflection_results.get("suggestions", "")
            previous_results = reflection_results.get("previous_relations", "")
            entity_extraction_results_text = reflection_results.get("previous_entities", "")
            score = reflection_results.get("score", "")
            
            # 添加agent提示
            agent_prompt_text = self.prompt_loader.render_prompt(
                'agent_prompt',
                variables={"abbreviations": abbreviations}
            )
            messages.append({"role": "system", "content": agent_prompt_text})
            
            # 添加背景信息
            background_info = ""
            if previous_suggestions:
                background_info += f"在进行关系提取的过程中，请特别关注：\n{previous_issues}\n相关问题的建议：\n{previous_suggestions}\n"
            
            if entity_extraction_results_text:
                background_info += f"在进行关系提取的过程中，请特别关注：\n{entity_extraction_results_text}\n"
            
            if background_info:
                messages.append({"role": "user", "content": background_info})
            
            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            
            logger.info("关系提取完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"关系提取过程中出现异常: {e}")
            error_result = {
                "error": f"关系提取失败: {str(e)}",
                "relations": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

