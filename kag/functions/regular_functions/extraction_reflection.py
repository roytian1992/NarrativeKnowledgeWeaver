from typing import Dict, Any, List
import json
import copy
from kag.utils.format import is_valid_json, correct_json_format

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


class ExtractionReflector:
    """抽取结果反思工具 - Qwen-Agent版本"""

    def __init__(self, prompt_loader=None, llm=None):
        super().__init__()
        self.prompt_loader = prompt_loader
        self.llm = llm

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
            logs = params_dict.get("logs", "")
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            relation_type_description_text = params_dict.get("relation_type_description_text", "")
            original_text = params_dict.get("original_text", "")
            abbreviations = params_dict.get("abbreviations", "")
            previous_reflection = params_dict.get("previous_reflection", {})
            
        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})

        if not logs:
            return json.dumps({"error": "缺少必要参数: logs"})

        try:
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='reflect_extraction_prompt',
                variables={
                    'logs': logs,
                    'entity_type_description_text': entity_type_description_text,
                    'relation_type_description_text': relation_type_description_text
                },
            )

            agent_prompt_text = self.prompt_loader.render_prompt(
                prompt_id="agent_prompt",
                variables={"abbreviations": abbreviations}
            )

            messages = [{"role": "system", "content": agent_prompt_text}]
            
            
            if original_text:
                original_info = f"这是之前信息抽取的原文：\n{original_text.strip()}" 
                messages.append({"role": "user", "content": original_info})
                
            messages.append({"role": "user", "content": f"这是实体和关系抽取时需要遵守的一些准则：\n{general_rules}"})
                    
            previous_issues = previous_reflection.get("issues", "")
            previous_suggestions = previous_reflection.get("suggestions", "")
            relation_extraction_results = previous_reflection.get("previous_relations", "")
            entitity_extraction_results = previous_reflection.get("previous_entities", "")
            score = previous_reflection.get("score", "")
            
            if previous_issues and score:
                summary = f"之前反思给出的得分为: {score}\n具体建议为：\n{previous_issues}\n，根据建议改进后的抽取结果为：\n\n"
                summary += f"实体抽取：\n{entitity_extraction_results}\n关系抽取:\n {relation_extraction_results}"
                # print("[CHECK] summary: ", summary)
                messages.append({"role": "user", "content": summary})
            
            
            messages.append({"role": "user", "content": prompt_text})
            starting_messages = messages.copy()
            
            full_response = ""
            max_round = 3

            for i in range(max_round):
                if i == 0:
                    enable_thinking=True
                else: 
                    enable_thinking=False
                result = self.llm.run(messages, enable_thinking=enable_thinking)
                content = result[0]['content']
                full_response += correct_json_format(content.strip())
                

                # 判断整体合法
                if is_valid_json(full_response):
                    return full_response

                # 添加补全提示
                messages.append({
                    "role": "assistant", "content": content
                })
                messages.append({
                    "role": "user",
                    "content": "请继续补全上一个 JSON 输出，禁止重复，直接继续输出 JSON 剩余部分："
                })

            # 最后修复尝试
            # repair_prompt = "你之前生成的 JSON 输出不完整或者格式错误，请修正它，确保返回合法、完整、符合 JSON 格式的结构，直接输出完整的正确的 JSON："
            # messages.append({"role": "assistant", "content": full_response})
            # messages.append({"role": "user", "content": repair_prompt})
            repair_result = self.llm.run(starting_messages, enable_thinking=True)
            full_response = repair_result[0]['content'].strip()

            if is_valid_json(full_response):
                return full_response
            else:
                print("[CHECK] 抽取反思失败: ", full_response)
                return json.dumps({
                    "error": "补全与修复尝试均失败，仍无法生成合法 JSON。",
                    "score": 0,
                    "current_issues": [],
                    "suggestions": []
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                    "error": f"抽取反思失败: {str(e)}",
                    "score": 0,
                    "current_issues": [],
                    "suggestions": []
                }, ensure_ascii=False)

