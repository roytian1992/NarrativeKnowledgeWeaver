from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下情节生成结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "plot_unit_id"字段，表示情节单元ID
2. "title"字段，表示情节标题
3. "description"字段，表示情节描述
4. "theme"字段，表示主题
5. "conflict"字段，表示冲突
6. "participants"字段，表示参与者列表
7. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


class PlotGenerator:
    """
    情节生成器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = [
            "plot_unit_id", "title", "description", "theme", 
            "conflict", "participants"
        ]
        self.field_validators = {
            "title": lambda x: isinstance(x, str) and len(x.strip()) > 0,
            "description": lambda x: isinstance(x, str) and len(x.strip()) > 10,
            "participants": lambda x: isinstance(x, list),
            "theme": lambda x: isinstance(x, str) and len(x.strip()) > 0,
            "conflict": lambda x: isinstance(x, str) and len(x.strip()) > 0
        }
        
        # 修复提示词模板
        self.repair_template = repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用情节生成，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数
            params_dict = json.loads(params)
            events = params_dict.get("events", [])
            cluster_info = params_dict.get("cluster_info", {})
            causality_threshold = params_dict.get("causality_threshold", "Medium")
            context = params_dict.get("context", "")
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {
                "error": f"参数解析失败: {str(e)}", 
                "plot_unit_id": "error_plot",
                "title": "错误情节",
                "description": f"参数解析失败: {str(e)}",
                "theme": "错误处理",
                "conflict": "系统异常",
                "participants": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not events:
            error_result = {
                "error": "缺少事件数据", 
                "plot_unit_id": "empty_plot",
                "title": "空情节",
                "description": "没有提供事件数据",
                "theme": "空内容",
                "conflict": "无冲突",
                "participants": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                "events": json.dumps(events, ensure_ascii=False, indent=2),
                "cluster_info": json.dumps(cluster_info, ensure_ascii=False, indent=2),
                "causality_threshold": causality_threshold,
                "context": context
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('plot_unit_construction_prompt', variables)
            
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
            
            logger.info("情节生成完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"情节生成过程中出现异常: {e}")
            error_result = {
                "error": f"情节生成失败: {str(e)}",
                "plot_unit_id": f"exception_plot_{cluster_info.get('cluster_id', 'unknown')}",
                "title": "异常情节单元",
                "description": f"生成过程中出现异常: {str(e)}",
                "theme": "异常处理",
                "conflict": "系统异常",
                "participants": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

