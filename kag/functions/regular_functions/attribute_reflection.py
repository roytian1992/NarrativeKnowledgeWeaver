from typing import Dict
import json
from kag.utils.format import is_valid_json, correct_json_format


class AttributeReflector:
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            entity_name = params_dict.get("entity_name", "")
            entity_type = params_dict.get("entity_type", "")
            description = params_dict.get("description", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")
            attributes = params_dict.get("attributes", "")
            abbreviations = params_dict.get("abbreviations", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False)

        if not entity_type or not description or not attribute_definitions or not attributes:
            print("[CHECK] 检查参数输入: ", params_dict)
            return json.dumps({"error": "缺少必要参数: entity_type / description / attribute_definitions / attributes"})

        try:
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="reflect_attributes_prompt",
                variables={
                    "entity_type": entity_type,
                    "description": description,
                    "entity_name": entity_name,
                    "attribute_definitions": attribute_definitions,
                    "attributes": attributes
                }
            )

            # Agent 提示词
            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={"abbreviations": abbreviations}
            )

            messages = [
                {"role": "system", "content": agent_prompt_text},
                {"role": "user", "content": prompt_text}
            ]
            starting_messages = messages.copy()
            # print("[CHECK] prompt text", prompt_text)

            full_response = ""
            max_round = 3

            for i in range(max_round):
                enable_thinking = (i == 0)
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                content = result[0]["content"]
                full_response += correct_json_format(content.strip())

                if is_valid_json(full_response):
                    return full_response

                # 请求补全 JSON
                messages.append({
                    "role": "user",
                    "content": "请继续补全上一个 JSON 输出，直接续写剩余部分，不要重复已有字段："
                })

            # 最后修复尝试
            # messages.append({
            #     "role": "user",
            #     "content": "你生成的 JSON 不完整或格式错误，请修正并完整输出："
            # })

            repair_result = self.llm.run(starting_messages, enable_thinking=True)
            full_response = repair_result[0]["content"].strip()

            if is_valid_json(full_response):
                return full_response
            else:
                return json.dumps({
                    "error": "反思失败，JSON 修复仍不合法。",
                    "feedbacks": [],
                    "score": 0,
                    "need_additional_context": "true"
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                    "error":  f"属性反思失败: {str(e)}",
                    "feedbacks": [],
                    "score": 0,
                    "need_additional_context": "true"
                }, ensure_ascii=False)
