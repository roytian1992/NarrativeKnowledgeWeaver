"""
场景角色状态抽取结果反思器（精简版）
仅使用提示词变量：scene_contents / extracted_results / character_list
"""
from typing import Any
import json
import logging

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class CharacterStatusReflector:
    """
    场景角色状态抽取结果反思器（精简版）

    - 仅依赖提示词变量：scene_contents / extracted_results / character_list
    - 对前一步的“角色状态抽取结果”进行内容质量审查（不检查 JSON 格式）
    - 输出结构固定为：
      {
        "feedbacks": ["问题或改进建议1", "问题或改进建议2", ...],
        "score": 0.0 ~ 10.0  // 内容质量评分
      }
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 必填字段与校验
        self.required_fields = ["feedbacks", "score"]
        self.field_validators = {
            "feedbacks": self._validate_feedbacks,
            "score": self._validate_score,
        }

        # 修复模板（若有专用模板可替换）
        self.repair_template = general_repair_template

    # -------------------- 校验函数 --------------------
    @staticmethod
    def _validate_feedbacks(value: Any) -> bool:
        """
        反馈应为字符串列表（允许为空列表）。
        """
        if not isinstance(value, list):
            return False
        return all(isinstance(x, str) for x in value)

    @staticmethod
    def _validate_score(value: Any) -> bool:
        """
        分数应为 0~10 的数字。
        """
        if not isinstance(value, (int, float)):
            return False
        return 0.0 <= float(value) <= 10.0

    # -------------------- 主调用入口 --------------------
    def call(self, params: str, enable_thinking: bool = True, **kwargs) -> str:
        """
        Args:
            params: JSON 字符串，必须包含：
              - scene_contents: str          原始场景文本
              - extracted_results: str       抽取结果的纯文本版本（你预先转换好的）
              - character_list: str 或 List[str]   参考角色列表（可选）
        Returns:
            str: 经过 correct_json_format 处理后的 JSON 字符串：
                 {
                   "feedbacks": [...],
                   "score": 0-10
                 }
        """
        try:
            params_dict = json.loads(params)
            scene_contents = params_dict.get("scene_contents", "")
            extracted_results = params_dict.get("extracted_results", "")
            character_list = params_dict.get("character_list", "")
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            error_result = {
                "error": f"参数解析失败: {str(e)}",
                "feedbacks": ["参数解析失败，无法进行质量反思。"],
                "score": 0.0,
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            # 渲染你定义的反思提示词（reflect_character_states_prompt）
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="reflect_character_status_prompt",
                variables={
                    "extracted_results": extracted_results,
                    "character_list": character_list,
                },
            )
            messages = []
            if scene_contents:
                # 如果有场景文本，就加进去（可选）
                messages.append({"role": "system", "content": f"当前任务的相关原文如下：\n{scene_contents}"})
            
            messages.append({"role": "user", "content": prompt_text})

            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template,
                enable_thinking=enable_thinking,
            )

            if status == "success":
                return corrected_json

            error_result = {
                "error": "场景角色状态抽取结果反思失败",
                "feedbacks": ["反思过程未成功完成，结果不可用。"],
                "score": 0.0,
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        except Exception as e:
            logger.error(f"角色状态反思过程中出现异常: {e}")
            error_result = {
                "error": f"场景角色状态抽取结果反思失败: {str(e)}",
                "feedbacks": ["反思过程出现异常，结果不可用。"],
                "score": 0.0,
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
