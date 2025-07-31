from typing import Dict, Any, List
import json
from kag.utils.format import is_valid_json, correct_json_format
import copy

general_rules = """
1. 禁止将实体类型当作实体：不要将 Concept、Event、Object、Action 等抽象描述错误地当作具体实体抽取。
2. 禁止关系主客不清或逻辑混乱：不应抽取语义含混、主客颠倒、方向不明或逻辑无法成立的关系。
3. 禁止使用非法类型值：关系类型字段必须严格使用系统提供的英文枚举值，禁止使用中文、拼音、自创词或大小写错误；不过关系名可以为具体的自然语言。
4. 仅在已识别实体之间抽取关系：关系抽取仅限于实体列表中已有的实体之间，禁止引入未列出的实体。
5. 无法明确关系时应放弃抽取：若无法判断实体之间是否存在明确关系，宁可不抽取，禁止猜测或强行生成。
6. 禁止抽取背景、无效或冗余实体：应忽略如浪花、海面、爆炸等背景元素，仅保留对叙事有意义的核心实体。
7. 忽略剧本辅助元素中的内容：字卡、视觉提示等辅助信息本身不可以作为实体被抽取出来，比如“解说（VO）”不能作为实体，但是里面包含的具体内容可以考虑。
8. 注意实体组合与主干识别：当实体名称中包含职称、称谓或修饰语（如“少校”、“老师”），应识别其核心指代对象为具体人物（如“刘培强”），而非将完整修饰短语（如“刘培强少校”）作为独立实体。
9. 每一个实体必须要有其相应的description。
"""


class EntityExtractor:

    def __init__(self, prompt_loader=None, llm=None):
        super().__init__()
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            # print("****", params_dict)
            text = params_dict.get("text", "")
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            abbreviations = params_dict.get("abbreviations", "")
            reflection_results = params_dict.get("reflection_results", {})
                        
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not text:
            return json.dumps({"error": "缺少必要参数: text"})

        try:
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='extract_entities_prompt',
                variables={
                    'text': text,
                    'entity_type_description_text': entity_type_description_text
                },
            )

            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={"abbreviations": abbreviations}
            )

           
            # 构造初始消息
            messages = [{"role": "system", "content": agent_prompt_text}]
            messages.append({"role": "user", "content": f"这是实体和关系抽取时需要遵守的一些准则：\n{general_rules}"})
            
            previous_issues = reflection_results.get("issues", [])
            previous_suggestions = reflection_results.get("suggestions", [])
            previous_results = reflection_results.get("previous_entities", "")
            score = reflection_results.get("score", "")
            
            background_info = ""
            if previous_suggestions:
                previous_suggestions = "\n".join(previous_suggestions)
                background_info +=  f"在执行知识图谱构建的过程中，以下是一些可供参考的建议：\n{previous_suggestions}\n\n"

            if previous_results and previous_issues and score:
                previous_iusses = "\n".join(previous_issues)
                background_info += f"这是你之前抽取的结果，部分内容有待改进： \n{previous_results}, 相关问题为: \n {previous_issues}，得分为: {score}"

            if background_info:
                messages.append({
                    "role": "user",
                    "content": background_info
                })

            messages.append({"role": "user", "content": prompt_text})
            starting_messages = messages.copy()
            
            full_response = ""
            max_round = 3

            for i in range(max_round):
                if i == 0:
                    enable_thinking = True
                else: 
                    enable_thinking = False
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                # print("[CHECK]: ", result)
                content = result[0]['content']
                full_response += correct_json_format(content.strip())

                if is_valid_json(full_response):
                    return full_response

                # 追加补全提示，而不是重建 messages
                messages.append({
                    "role": "user",
                    "content": "请继续补全上一个 JSON 输出，禁止重复已经有的实体和内容，直接继续输出 JSON 剩余部分："
                })

            # 最后一次尝试修复
            # repair_prompt = "你之前生成的 JSON 输出不完整或者格式错误，请修正它，确保返回合法、完整、符合 JSON 格式的结构，直接输出完整的正确的 JSON："
            # messages.append({
            #     "role": "user",
            #     "content": repair_prompt
            # })

            repair_result = self.llm.run(starting_messages, enable_thinking=True)
            full_response = repair_result[0]['content'].strip()

            if is_valid_json(full_response):
                return full_response
            else:
                return json.dumps({
                    "error": "补全与修复尝试均失败，仍无法生成合法 JSON。",
                    "entities": []
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                    "error": f"实体抽取失败: {str(e)}",
                    "entities": []
                }, ensure_ascii=False)
