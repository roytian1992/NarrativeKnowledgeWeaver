from typing import Dict
import json
from kag.utils.format import is_valid_json, correct_json_format


class AttributeExtractor:
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            entity_name = params_dict.get("entity_name", "")
            description = params_dict.get("description", "")
            entity_type = params_dict.get("entity_type", "")
            attribute_definitions = params_dict.get("attribute_definitions", "")
            abbreviations = params_dict.get("abbreviations", "")  # 和实体抽取逻辑保持一致
            feedbacks = params_dict.get("feedbacks", "")
            original_text = params_dict.get("original_text", "")
            previous_results = params_dict.get("previous_results", "")

        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False)

        if not text or not description or not entity_type:
            print("[CHECK] 检查参数输入: ", params_dict)
            return json.dumps({"error": "缺少必要参数: text/description/entity_type"}, ensure_ascii=False)

        try:
            # 主抽取提示词
            if original_text and previous_results and feedbacks:
                text = f"这些是之前的上下文：\n{original_text } \n这些是新增的文本，用于对已有的抽取结果进行补充和改进:\n{text}\n"
                
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_attributes_prompt',
                variables={
                    "text": text,
                    "entity_name": entity_name,
                    "description": description,
                    "entity_type": entity_type,
                    "attribute_definitions": attribute_definitions
                }
            )
            
            # agent 指令（system prompt），同你之前写法
            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={"abbreviations": abbreviations}
            )
            messages = [{"role": "system", "content": agent_prompt_text}]
            
            if original_text and previous_results and feedbacks:
                background_info = f"上一次信息抽取的上下文：\n{original_text.strip()}\n\n" 
                
                background_info += f"上一次抽取的结果如下：\n{previous_results}\n反馈建议如下：\n{feedbacks}\n请仅针对缺失字段或内容不足的字段进行补充，保留已有字段。"
                
                messages.append({
                    "role": "user",
                    "content": background_info
                })
                
                prompt_text = prompt_text + "\n" + f"这是之前抽取的结果：\n {previous_results} \n 在此基础上根据建议进行补充和改进。"

            messages.append({"role": "user", "content": prompt_text})

            starting_messages = messages.copy()
            # print("[CHECK] prompt text", prompt_text)
            full_response = ""
            max_round = 2

            for i in range(max_round):
                enable_thinking = (i == 0)
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                content = result[0]["content"]
                full_response += correct_json_format(content.strip())

                if is_valid_json(full_response):
                    return full_response

                # 增量补全请求
                messages.append({
                    "role": "user",
                    "content": "请继续补全上一个 JSON 输出，直接续写剩余部分，不要重复已有字段："
                })

            # 最后修复尝试
            # messages.append({
            #     "role": "user",
            #     "content": "你生成的 JSON 不完整或格式错误，请输出修正后的完整 JSON："
            # })

            repair_result = self.llm.run(starting_messages, enable_thinking=True)
            full_response = repair_result[0]["content"].strip()

            if is_valid_json(full_response):
                return full_response
            else:
                return json.dumps({
                    "error": "补全与修复尝试失败，JSON 无效。",
                    "partial_result": full_response
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"属性抽取失败: {str(e)}"}, ensure_ascii=False)
