# 比较易于编辑的提示词模板
import os
import json


def generate_agent_prompt(path):
    template = """你是一个专业的知识图谱构建专家，擅长从文本中抽取实体和关系。
你的任务是：
1. 识别文本中的重要实体（人物、地点、物品、概念、时间等）
2. 分析实体间的关系（家庭关系、社会关系、空间关系、时间关系等）
3. 对于剧本文本，特别关注场景、对话、角色关系等特有元素

请使用提供的工具来完成信息抽取任务。"""

    prompt_template = {
    "id": "agent_prompt",
    "category": "agent",
    "name": "Agent提示",
    "description": "用于指导Agent的行为",
    "template": template,
    "variables": []
    }

    path = os.path.join(path, "agent_prompt.json")
    with open(path, "w") as json_file:
        json.dump(prompt_template, json_file, ensure_ascii=False, indent=2)


def generate_extract_entities_tool_prompt(path):
    # extract_entities_tool_prompt.json

    template = """从以下文本中识别并抽取实体，按照以下格式严格返回：

**实体类型 (type 字段) 只能使用以下英文枚举值，禁止自定义、翻译或扩展，禁止使用中文类型名：**

{entity_type_description_text}

**严格要求：**
- type 字段必须完全等于上述枚举值之一，大小写敏感。
- 如果不确定实体类型，默认使用 Concept。
- 禁止生成非枚举值、拼音、中文类型名、首字母小写、驼峰拼写等错误。

文本内容：
{text}

请按以下严格格式返回 JSON：
```json
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型 (英文枚举值之一)",
      "description": "实体描述",
      "aliases": ["别名1", "别名2"]
    }
  ]
...
**注意：**
- 仅输出符合 JSON 格式的内容，禁止添加额外解释、注释、自然语言说明。
- type 字段必须严格匹配英文枚举值，不能生成其他值。
- 若无别名，可返回空数组 aliases: []."""


    prompt_template = {
    "id": "extract_entities_tool_prompt",
    "category": "tool",
    "name": "实体抽取工具提示",
    "description": "用于提示LLM进行实体抽取",
    "template": template,
    "variables": [
        {
        "name": "text",
        "description": "待抽取的文本内容"
        },
        {
        "name": "entity_type_description_text",
        "description": "实体类型列表（自动生成，英文枚举值 + 描述，换行分隔）"
        }
    ]
    }

    path = os.path.join(path, "extract_entites_tool_prompt.json")
    with open(path, "w") as json_file:
        json.dump(prompt_template, json_file, ensure_ascii=False, indent=2)


def generate_extract_relations_tool_prompt(path):
    # extract_relations_tool_prompt.json

    template = """从以下文本中识别实体间的关系。

**只能使用以下关系类型枚举值，禁止使用自然语言或其他未定义关系类型！**

{relation_type_description_text}

**禁止自创关系类型，禁止使用中文关系名，关系字段必须严格对应上方英文枚举值之一。**

输出格式严格如下（字段名必须一致，禁止修改字段名）：
```json
{
  "relations": [
    {
      "subject": "主体实体",
      "predicate": "关系类型 (上述英文枚举值之一)",
      "object": "客体实体",
      "description": "可选的关系描述（可以为空字符串）"
    }
  ]
}
```

**注意：**
- 仅输出符合 JSON 格式的内容，禁止额外添加注释或自然语言。
- 禁止输出 JSON 之外的文本。

文本内容：
{text}"""


    prompt_template = {
    "id": "extract_relations_tool_prompt",
    "category": "tool",
    "name": "关系抽取工具提示",
    "description": "用于提示LLM进行关系抽取",
    "template": "从以下文本中识别实体间的关系。\n\n**只能使用以下关系类型枚举值，禁止使用自然语言或其他未定义关系类型！**\n\n{relation_type_description_text}\n\n**禁止自创关系类型，禁止使用中文关系名，关系字段必须严格对应上方英文枚举值之一。**\n\n输出格式严格如下（字段名必须一致，禁止修改字段名）：\n```json\n{\n  \"relations\": [\n    {\n      \"subject\": \"主体实体\",\n      \"predicate\": \"关系类型 (上述英文枚举值之一)\",\n      \"object\": \"客体实体\",\n      \"description\": \"可选的关系描述（可以为空字符串）\"\n    }\n  ]\n}\n```\n\n**注意：**\n- 仅输出符合 JSON 格式的内容，禁止额外添加注释或自然语言。\n- 禁止输出 JSON 之外的文本。\n\n文本内容：\n{text}",
    "variables": [
        {
        "name": "text",
        "description": "待抽取的文本内容"
        },
        {
        "name": "relation_type_description_text",
        "description": "关系类型列表（自动生成，英文枚举值 + 描述，换行分隔）"
        }
    ]
    }

    path = os.path.join(path, "extract_relations_tool_prompt.json")
    with open(path, "w") as json_file:
        json.dump(prompt_template, json_file, ensure_ascii=False, indent=2)


def generate_extract_script_elements_tool_prompt(path):
    # extract_script_elements_tool_prompt.json

    template = """从以下剧本文本中抽取特有元素，按要求填写JSON：

**注意事项：**
- 场景信息重点补充详细描述（地点、时间、环境），不重复列出已有实体（Location、Time、Scene）。
- 角色对话只提取关键性对话，保持完整原文句子。
- 事件补充关键情节点，不与已有Event实体重复。

文本内容：
{text}

请按以下严格格式返回JSON（字段名不可修改）：
```json
{
"script_elements": {
    "scenes": ["场景描述1", "场景描述2"],
    "characters": ["角色名称1", "角色名称2"],
    "events": ["关键事件1", "关键事件2"],
}
}
```
...
- 输出JSON之外的自然语言或注释。
- 修改字段名、层级结构。

若某一项无内容，请返回空数组，如 "characters": []."""


    prompt_template = {
    "id": "extract_scene_elements_tool_prompt",
    "category": "tool",
    "name": "场景元素抽取工具提示",
    "description": "用于提示LLM抽取剧本文本特有元素",
    "template": template,
    "variables": [
        {
        "name": "text",
        "description": "待抽取的剧本文本内容"
        }
    ]
    }

    path = os.path.join(path, "extract_script_elements_tool_prompt.json")
    with open(path, "w") as json_file:
        json.dump(prompt_template, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    path = "./"
    generate_agent_prompt(path)
    generate_extract_entities_tool_prompt(path)
    generate_extract_relations_tool_prompt(path)
    generate_extract_script_elements_tool_prompt(path)