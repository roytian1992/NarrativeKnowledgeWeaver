from typing import Dict, Any
import json
from kag.utils.format import is_valid_json, correct_json_format


class GraphReflector:
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            subject_name = params_dict["subject_name"]
            predicate = params_dict["predicate"]
            object_name = params_dict["object_name"]
            entity_name = params_dict["entity_name"]
            relation_name = params_dict["relation_name"]
            description = params_dict["description"]
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        # 渲染提示词
        prompt_text = self.prompt_loader.render_prompt(
            prompt_id="reflect_graph_prompt",
            variables={
                "subject_name": subject_name,
                "predicate": predicate,
                "object_name": object_name,
                "entity_name": entity_name,
                "relation_name": relation_name,
                "description": description
            }
        )

        messages = [{"role": "user", "content": prompt_text}]

        # 最多重试 3 次
        for attempt in range(3):
            try:
                result = self.llm.run(messages, enable_thinking=False)
                content = result[0]["content"].strip()
                content = correct_json_format(content)

                if is_valid_json(content):
                    return content
            except Exception as e:
                last_exception = str(e)

        fallback = json.dumps({
            "error": "生成结果不是合法 JSON 格式，返回默认判断结果",
            "raw_output": content if 'content' in locals() else None,
            "keep_relation": False,
            "entities_to_check": []
        }, ensure_ascii=False)
        print("[CHECK] fallback:", fallback)
        return fallback
