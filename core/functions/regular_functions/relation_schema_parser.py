from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class RelationSchemaParser:
    """
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["relations"]
        self.field_validators = {}
        
        # 修复提示词模板
        self.repair_template = general_repair_template
    
    def call(self, params: str, **kwargs) -> str:
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
            entity_schema = params_dict.get("entity_chema", "")
            current_schema = params_dict.get("current_schema", "")
            feedbacks = params_dict.get("feedbacks", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "relations":{}}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
        try:
            # 构造初始消息
            messages = []
            background_info = ""
            if current_schema:
                background_info += f"这是当前使用的schema：\n{current_schema}\n请在后续的任务中，基于这个进行调整。"
                if feedbacks:
                    background_info += f"\n这是针对当前schema的一些建议：\n{feedbacks}"
            
            if not background_info:
                background_info = "无"
                
            messages.append({"role": "user", "content": f"以下是阅读时的一些相关的洞见：\n{text}"})
                
            prompt_id = "parse_relation_schema_prompt"
            variables = {  
                "current_background": background_info,
                "entity_schema": entity_schema        
            }
                
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id=prompt_id,
                variables=variables
            )
            # print("[CHECK] prompt_text: ", prompt_text)

            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=2,
                repair_template=self.repair_template
            )
            if status == "success":
                logger.info("relation schema提取完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"relation schema生成失败",
                    "relations":{}
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"relation schema提取过程中出现异常: {e}")
            error_result = {
                "error": f"relation schema生成失败: {str(e)}",
                "relations":{}
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

