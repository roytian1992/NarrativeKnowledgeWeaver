"""
增量属性提取器
仅通过提示词模板输入所有信息，不做额外文案包装
"""
from typing import Dict, Any
import json
import logging

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import attribute_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class AttributeUpdater:
    """
    增量属性提取器

    - 使用 prompt_id='incremental_extract_attributes_prompt'
    - 所有上下文（原文、prev_attributes、prev_description、feedbacks 等）
      都只通过模板变量传进去，**不额外再拼别的 user 文本**
    - 返回结构固定：
      {
        "new_description": "...",
        "attributes": { ... }
      }
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 要求至少这两个字段
        self.required_fields = ["attributes", "new_description"]
        self.field_validators = {
            "attributes": lambda x: isinstance(x, dict),
            "new_description": lambda x: isinstance(x, str) and x.strip() != "",
        }

        self.repair_template = attribute_repair_template

    def call(self, params: str, enable_thinking: bool = True, **kwargs) -> str:
        """
        Args:
            params: JSON 字符串，推荐结构（但不强制）：
                {
                  "text": "... 当前 chunk 文本 ...",
                  "entity_name": "...",
                  "entity_type": "...",
                  "description": "实体类型描述",
                  "attribute_definitions": "... 或 dict/list ...",
                  "system_prompt": "...",
                  "prev_attributes": {...} 或 JSON 字符串，可选,
                  "prev_description": "...", 可选
                  "feedbacks": "... 可选"
                }

        Returns:
            str: correct_json_format 之后的 JSON 字符串：
                {
                  "new_description": "...",
                  "attributes": { ... }
                }
        """
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            error_result = {
                "error": f"参数解析失败: {str(e)}",
                "new_description": "",
                "attributes": {},
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            text = params_dict.get("text", "")
            entity_name = params_dict.get("entity_name", "")
            description = params_dict.get("description", "")
            entity_type = params_dict.get("entity_type", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")

            system_prompt = params_dict.get("system_prompt", "")

            prev_attributes = params_dict.get("prev_attributes", "")
            prev_description = params_dict.get("prev_description", "")
            feedbacks = params_dict.get("feedbacks", "")

            # -------- 把可能是 dict/list 的字段统一转成字符串，方便模板渲染 --------
            def _to_text(v) -> str:
                if isinstance(v, (dict, list)):
                    return json.dumps(v, ensure_ascii=False, indent=2)
                return "" if v is None else str(v)

            attribute_definitions_text = _to_text(attribute_definitions)
            prev_attributes_text = _to_text(prev_attributes)
            prev_description_text = _to_text(prev_description)
            feedbacks_text = _to_text(feedbacks)

            # -------- 只通过模板变量把所有东西塞进提示词 --------
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="update_attributes_prompt",
                variables={
                    "text": text,
                    "entity_name": entity_name,
                    "description": description,
                    "entity_type": entity_type,
                    "attribute_definitions": attribute_definitions_text,
                    "prev_attributes": prev_attributes_text,
                    "prev_description": prev_description_text,
                    "feedbacks": feedbacks_text,
                },
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 只保留这一条 user 消息，由模板负责组织所有内容
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
            else:
                error_result = {
                    "error": "增量属性提取失败",
                    "new_description": "",
                    "attributes": {},
                }
                return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        except Exception as e:
            logger.error(f"增量属性提取过程中出现异常: {e}")
            error_result = {
                "error": f"增量属性提取失败: {str(e)}",
                "new_description": "",
                "attributes": {},
            }
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
