from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

class EntitySchemaParser:
    """
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["entities"]
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
            current_schema = params_dict.get("current_schema", "")
            feedbacks = params_dict.get("feedbacks", "")
            task_goals = params_dict.get("task_goals", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "entities":{}}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
                
        try:
            # 构造初始消息
            messages = []
            background_info = ""
            if task_goals:
                background_info += f"你是一个构建知识图谱的专家，请你基于以下一些任务（仅供参考，非硬性要求），思考一下我们的图谱所需要的schema：\n{task_goals}"
            
            if current_schema:
                background_info += f"这是当前使用的schema：\n{current_schema}\n请在后续的任务中，基于这个进行调整。如果已经有的description尽量不要进行删减，只进行增量补充。"
                if feedbacks:
                    background_info += f"\n这是针对当前schema的一些建议：\n{feedbacks}"
            if not background_info:
                background_info = "无"
                
            messages.append({"role": "user", "content": f"以下是阅读时的一些相关的洞见：\n{text}"})
                
            prompt_id = "parse_entity_schema_prompt"
            variables = {"current_background": background_info}
                
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
                # logger.info("entity schema提取完成，返回格式化后的JSON")
                return corrected_json
            else:
                error_result = {
                    "error": f"entity schema提取失败",
                    "entities":{}
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"entity schema提取过程中出现异常: {e}")
            error_result = {
                "error": f"entity schema提取失败: {str(e)}",
                "entities":{}
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

