general_rules = """
1. 禁止将实体类型当作实体：不要将 Concept、Event、Object、Action 等实体类型错误地当作具体实体抽取；也不要把“你”、“我”、“他”这些代词当做实体名称。
2. 禁止关系主客不清或逻辑混乱：不应抽取语义含混、主客颠倒、方向不明或逻辑无法成立的关系。
3. 禁止使用非法类型值：关系类型字段必须严格使用系统提供的英文枚举值，禁止使用中文、拼音、自创词或大小写错误；不过关系名可以为具体的自然语言。
4. 仅在已识别实体之间抽取关系：关系抽取仅限于实体列表中已有的实体之间，禁止引入未列出的实体。
5. 无法明确关系时应放弃抽取：若无法判断实体之间是否存在明确关系，宁可不抽取，禁止猜测或强行生成。
6. 禁止抽取背景、无效或冗余实体：忽略如浪花、海面、爆炸等背景元素，仅保留对叙事有意义的核心实体。
7. 忽略剧本辅助元素中的内容：字卡、视觉提示等辅助信息本身不可以作为实体被抽取出来，比如“解说（VO）”不能作为实体，但是里面包含的具体内容可以考虑。
8. 注意实体组合与主干识别：当实体名称中包含职称、称谓或修饰语（如“少校”、“先生”、“小姐”、“女士”、“老师”、“博士”），应识别其核心指代对象为具体人物（如“刘培强”），而非将完整修饰短语（如“刘培强少校”）作为独立实体。
9. 每一个实体必须要有其相应的description。实体可能属于多个类型，反思的时候如果原先的抽取也有道理，不必苛求修正。
"""

general_repair_template = """
请修复以下响应中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含格式正确。

请直接返回修复后的JSON，不要包含解释。
"""


semantic_splitter_repair_template = """
请修复以下文本分割结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "segments"字段
2. 每个分段包含必要的字段信息

建议：
1. 认真检查引号等问题，确保JSON格式正确。
2. 如果有标点符号的问题，请帮我改正。
3. 必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""


entity_repair_template = """
请修复以下实体提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "entities"字段，且为非空数组
2. 每个实体包含"name"和"type"字段

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""

relation_repair_template = """
请修复以下关系提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "relations"字段，且为数组格式
2. 每个关系包含必要的字段信息

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""

extraction_refletion_repair_template = """
请修复以下提取反思结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "score"字段，包含最终评分
2. "current_issues"字段，包含发现的问题列表
3. "suggestions"字段，包含改进建议列表

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""

attribute_repair_template = """
请修复以下属性提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "attributes"字段，且为数组格式
2. 每个属性包含必要的字段信息

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""

attribute_reflection_repair_template = """
请修复以下属性反思结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "feedbacks"字段，表示当前的反馈
2. "need_additional_context"字段，表示是否需要额外的信息
3. "attributes_to_retry"字段，表示需要继续抽取的属性

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。
"""

causality_check_repair_template = """
请修复以下因果关系检查结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "causal"字段，表示因果关系结果 (High/Medium/Low/None)
2. "reason"字段，表示原因说明，
3. "confidence"字段，表示置信度。

必要的时候可以直接原始响应抽取字段里面的信息进行重组。

请直接返回修复后的JSON，不要包含解释。必要的时候可以直接抽取字段里面的信息
"""