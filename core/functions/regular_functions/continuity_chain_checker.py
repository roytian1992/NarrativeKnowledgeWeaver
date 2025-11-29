"""
接戏链判定器
使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from core.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee
from core.utils.general_text import attribute_repair_template  # 复用通用repair模板
from core.utils.format import correct_json_format, is_valid_json

logger = logging.getLogger(__name__)


class ContinuityChainChecker:
    """
    接戏链（多场景）制作连续性判定器

    对一条自动生成的接戏链（多个 scene 的列表）进行整体评估：
    - 连贯度评分 coherence_score
    - 决策 keep / split / drop
    - 若 split 则给出 suggested_splits
    - rationale & decision_confidence

    确保最终返回的是经过 correct_json_format 处理后的 JSON 字符串。
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 必须字段
        self.required_fields = [
            "coherence_score",
            "keep_decision",
            "suggested_splits",
            "rationale",
            "decision_confidence",
        ]

        # 字段校验规则
        self.field_validators = {
            "coherence_score": self._validate_coherence_score,
            "keep_decision": self._validate_keep_decision,
            "suggested_splits": self._validate_suggested_splits,
            "rationale": lambda x: isinstance(x, str),
            "decision_confidence": self._validate_decision_confidence,
        }

        # 通用修复模板
        self.repair_template = attribute_repair_template

    # ====== 字段校验方法 ======
    @staticmethod
    def _validate_coherence_score(value: Any) -> bool:
        try:
            v = float(value)
            return 0.0 <= v <= 1.0
        except Exception:
            return False

    @staticmethod
    def _validate_keep_decision(value: Any) -> bool:
        return value in ["keep", "split", "drop"]

    @staticmethod
    def _validate_decision_confidence(value: Any) -> bool:
        return value in ["low", "medium", "high"]

    @staticmethod
    def _validate_suggested_splits(value: Any) -> bool:
        # 期望：二维数组，例如 [["scene_3", "scene_5"], ["scene_10"]]
        if not isinstance(value, list):
            return False
        for group in value:
            if not isinstance(group, list):
                return False
            # 元素可以是字符串或者可转成字符串的东西，这里只做类型检查
            for sid in group:
                if not isinstance(sid, (str, int)):
                    return False
        return True

    def call(self, params: str, enable_thinking: bool = True, **kwargs) -> str:
        """
        调用接戏链判定，保证返回 correct_json_format 处理后的结果

        Args:
            params: 参数字符串，期望为 JSON，包含：
                - scene_id_list: 按剧本顺序排列的场景 ID 列表（或字符串）
                - scene_metadata_blocks: 每个场景的元信息摘要文本

        Returns:
            str: 经过 correct_json_format 处理的 JSON 字符串：
                {
                  "coherence_score": 0.0-1.0,
                  "keep_decision": "keep" | "split" | "drop",
                  "suggested_splits": [...],
                  "rationale": "简短说明",
                  "decision_confidence": "low" | "medium" | "high"
                }
        """
        try:
            # 解析参数
            params_dict = json.loads(params)

            scene_id_list = params_dict.get("scene_id_list", "")
            scene_metadata_blocks = params_dict.get("scene_metadata_blocks", "")

            # 容错：如果 scene_id_list 是 list，就转成逗号分隔字符串
            if isinstance(scene_id_list, list):
                scene_id_list = ", ".join(str(s) for s in scene_id_list)

        except Exception as e:
            logger.error(f"接戏链参数解析失败: {e}")
            error_result = {
                "error": f"参数解析失败: {str(e)}",
                "coherence_score": 0.0,
                "keep_decision": "keep",
                "suggested_splits": [],
                "rationale": "",
                "decision_confidence": "low"
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            # 渲染接戏链判定提示词（注意这里的 prompt_id 要和你 JSON 里的一致）
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="continuity_chain_check_prompt",
                variables={
                    "scene_id_list": scene_id_list,
                    "scene_metadata_blocks": scene_metadata_blocks,
                },
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ]

            # 使用增强工具处理响应，保证最终 JSON 完整&结构正确
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
                # corrected_json 已经是 correct_json_format 处理后的字符串
                return corrected_json
            else:
                error_result = {
                    "error": "接戏链判定失败",
                    "coherence_score": 0.0,
                    "keep_decision": "keep",
                    "suggested_splits": [],
                    "rationale": "",
                    "decision_confidence": "low"
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        except Exception as e:
            logger.error(f"接戏链判定过程中出现异常: {e}")
            error_result = {
                "error": f"接戏链判定失败: {str(e)}",
                "coherence_score": 0.0,
                "keep_decision": "keep",
                "suggested_splits": [],
                "rationale": "",
                "decision_confidence": "low"
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
