"""
接戏判定器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import attribute_repair_template  # 这里复用通用repair模板
from core.utils.format import correct_json_format, is_valid_json

logger = logging.getLogger(__name__)

class ContinuityChecker:
    """
    接戏（连戏）关系判定器
    确保最终返回的是 correct_json_format 处理后的结果
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 定义验证规则：必须包含 is_continuity 和 reason 字段
        self.required_fields = ["is_continuity", "reason"]
        self.field_validators = {
            "is_continuity": lambda x: isinstance(x, bool),
            "reason": lambda x: isinstance(x, str)
        }

        # 这里复用 attribute_repair_template 作为通用修复提示词模板
        self.repair_template = attribute_repair_template

    def call(self, params: str, enable_thinking: bool = True, **kwargs) -> str:
        """
        调用接戏判定，保证返回 correct_json_format 处理后的结果

        Args:
            params: 参数字符串，期望为JSON，包含：
                - scene_name1, summary1, cmp_info1
                - scene_name2, summary2, cmp_info2
                - common_neighbor_info

        Returns:
            str: 经过 correct_json_format 处理的 JSON 字符串：
                {
                  "is_continuity": true/false,
                  "reason": "解释理由"
                }
        """
        try:
            # 解析参数
            params_dict = json.loads(params)

            scene_name1 = params_dict.get("scene_name1", "")
            summary1 = params_dict.get("summary1", "")
            cmp_info1 = params_dict.get("cmp_info1", "")

            scene_name2 = params_dict.get("scene_name2", "")
            summary2 = params_dict.get("summary2", "")
            cmp_info2 = params_dict.get("cmp_info2", "")

            common_neighbor_info = params_dict.get("common_neighbor_info", "")

        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过 correct_json_format 处理
            error_result = {
                "error": f"参数解析失败: {str(e)}",
                "is_continuity": False,
                "reason": ""
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            # 渲染接戏判定提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='continuity_check_prompt',
                variables={
                    "scene_name1": scene_name1,
                    "summary1": summary1,
                    "cmp_info1": cmp_info1,
                    "scene_name2": scene_name2,
                    "summary2": summary2,
                    "cmp_info2": cmp_info2,
                    "common_neighbor_info": common_neighbor_info
                }
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]

            # 使用增强工具处理响应，保证返回 correct_json_format 处理后的结果
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template,
                enable_thinking=enable_thinking
            )

            if status == "success":
                return corrected_json
            else:
                error_result = {
                    "error": "接戏判定失败",
                    "is_continuity": False,
                    "reason": ""
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        except Exception as e:
            logger.error(f"接戏判定过程中出现异常: {e}")
            error_result = {
                "error": f"接戏判定失败: {str(e)}",
                "is_continuity": False,
                "reason": ""
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
