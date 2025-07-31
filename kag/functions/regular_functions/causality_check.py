from typing import Dict, Any
import json
from kag.utils.format import is_valid_json, correct_json_format


class EventCausalityChecker:
    """
    事件因果性判断工具：输入两个事件的描述，判断其是否存在因果关系。
    输出 JSON 格式：
    {
        "causal": "High / Medium / Low / None",
        "reason": "简要说明判断依据"
    }
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            event_1_info = params_dict.get("event_1_info", "")
            event_2_info = params_dict.get("event_2_info", "")
            abbreviations = params_dict.get("abbreviations", "")
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not event_1_info or not event_2_info:
            return json.dumps({"error": "缺少必要参数: event_1_info 或 event_2_info"})

        try:
            # 构造提示词
            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={"abbreviations": abbreviations}
            )
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="check_event_causality_prompt",
                variables={
                    "event_1_info": event_1_info,
                    "event_2_info": event_2_info
                }
            )

            messages = [
                {"role": "system", "content": agent_prompt_text},
                {"role": "user", "content": prompt_text}
            ]
            starting_messages = messages.copy()

            full_response = ""
            max_round = 3

            for i in range(max_round):
                enable_thinking = (i == 0)
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                content = result[0]["content"].strip()
                full_response += correct_json_format(content)

                if is_valid_json(full_response):
                    return full_response

                messages.append({
                    "role": "user",
                    "content": "你刚才的 JSON 输出不完整或格式有误，请继续补全剩余部分，直接续写，不要重复已有字段："
                })

            # 最后一轮重试：重新执行起始 prompt（带思考）
            repair_result = self.llm.run(starting_messages, enable_thinking=True)
            final_response = repair_result[0]["content"].strip()
            formatted = correct_json_format(final_response)

            if is_valid_json(formatted):
                return formatted
            else:
                return json.dumps({
                    "error": "多轮修复失败，输出仍不合法。",
                    "causal": "Low",
                    "reason": "抽取失败，不可信结果"
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"检查因果关系失败: {str(e)}",
                "causal": "Low",
                "reason": "抽取失败，不可信结果"
            }, ensure_ascii=False)
