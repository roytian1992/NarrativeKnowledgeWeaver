from typing import Dict, Any
import json
from kag.utils.format import is_valid_json, correct_json_format


class SemanticSplitter:

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            max_segments = params_dict.get("max_segments", 3)
            min_length = params_dict.get("min_length", len(text) * 0.4)
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not text:
            return json.dumps({"error": "缺少必要参数: text"})

        prompt_text = self.prompt_loader.render_prompt(
            prompt_id="semantic_splitter_prompt",
            variables={
                "text": text,
                "min_length": min_length,
                "max_segments": max_segments
            }
        )

        messages = [{"role": "user", "content": prompt_text}]
        starting_messages = messages.copy()

        # 最多重试 3 次
        for attempt in range(3):
            try:
                if attempt == 0:
                    enable_thinking = True
                else:
                    enable_thinking = False
                    
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                content = result[0]['content'].strip()
                content = correct_json_format(content)

                if is_valid_json(content):
                    return content
            except Exception as e:
                last_exception = str(e)

        # 最后一次尝试修复
        result = self.llm.run(messages, enable_thinking=enable_thinking)
        content = result[0]['content'].strip()
        content = correct_json_format(content)

        if is_valid_json(content):
            return content
        else:
            # 所有尝试失败，返回 fallback 格式
            fallback = json.dumps({
                "error": "生成结果不是合法 JSON 格式，已返回原始文本作为唯一分段",
                "raw_output": content if 'content' in locals() else None,
                "segments": [text, ""]
            }, ensure_ascii=False)
            print("[CHECK] fallback", fallback)
            return fallback
