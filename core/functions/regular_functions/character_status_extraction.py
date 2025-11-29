"""
场景角色时序状态抽取器（精简版）
仅使用提示词变量：scene_contents / timelines / character_list
"""
from typing import Any
import json
import logging
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class CharacterStatusExtractor:
    """
    场景角色时序状态抽取器（精简版）
    - 仅依赖提示词变量：scene_contents / timelines / character_list
    - 输出结构固定为：
      {
        "timelines": [
          { "time": "...", "characters": [ { "name": "...", "status": "..." } ] }
        ]
      }
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 必填字段与校验
        self.required_fields = ["results"]
        self.field_validators = {}

        # 修复模板（若有专用模板可替换）
        self.repair_template =  general_repair_template

    def call(self, params: str, enable_thinking=True, **kwargs) -> str:
        """
        Args:
            params: JSON 字符串，必须包含：
              - scene_contents: str
              - timelines: str（可空/缺失）
              - character_list: str 或 List[str]
        Returns:
            str: 经过 correct_json_format 处理后的 JSON 字符串
        """
        try:
            params_dict = json.loads(params)
            scene_contents = params_dict.get("scene_contents", "")
            character_list = params_dict.get("character_list", "")
            feedbacks = params_dict.get("feedbacks", "")
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            error_result = {"error": f"参数解析失败: {str(e)}", "timelines": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            # 渲染与你提供的提示词（extract_character_states_prompt）一致的模板
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="extract_character_status_prompt",
                variables={
                    "scene_contents": scene_contents,
                    "character_list": character_list
                }
            )
            # print(prompt_text)
            # 精简消息，仅一条 user 内容
            messages = []
            if feedbacks:
                messages.append({"role": "system", "content": f"以下是对你之前抽取结果的反馈，请参考改进：\n{feedbacks}"})
            messages.append({"role": "user", "content": prompt_text})

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

            error_result = {"error": "场景角色时序状态抽取失败", "timelines": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        except Exception as e:
            logger.error(f"时序状态抽取过程中出现异常: {e}")
            error_result = {"error": f"场景角色时序状态抽取失败: {str(e)}", "timelines": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
