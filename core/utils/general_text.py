general_rules = """
1. 实体类型识别时，不应将 Concept、Event、Object、Action 等类别名称、人称代词（如“我”“你”“他”）以及“母亲”“父亲”等缺乏明确指向的称谓作为实体名称，只有在具备清晰指代（如“张三的母亲”）时才可作为实体。
2. 关系语义明确：抽取的关系需主客分明、语义清晰、方向明确，逻辑成立。
3. 关系类型合法：关系类型字段必须严格使用系统提供的英文枚举值，禁止使用其他形式；关系名称可为自然语言。
4. 关系实体来源：仅在已识别的实体之间抽取关系，不引入额外未识别的实体。
5. 抽取决策：无法明确判断的关系应放弃抽取，避免猜测或凭空生成。
6. 核心实体优先：仅保留与叙事推进或核心情节相关的实体，忽略纯背景性元素（如浪花、海面、爆炸）。
7. 辅助信息处理：剧本中的字卡、视觉提示等辅助信息本身不作为实体，但其中包含的具体内容可视情况抽取。
8. 实体组合识别：当实体名称包含称谓、职称或修饰语（如“少校”、“先生”、“博士”），应识别其核心人物（如“刘培强”），而非完整修饰短语。
9. 实体描述：每个实体必须提供 description 字段；若原有抽取合理，反思时不必苛求修改。
10. 语义范围（scope）：仅可为 "global" 或 "local"。具备明确身份、具名且跨章节/场景复现的为 "global"（如具体的出现名字的人物、概念，例如：田秋雨、假田秋雨）；一次性、泛指且依赖上下文的为 "local"（如“一名研究员”、“一位乘客”、“一个箱子”）。
11. 事件抽取原则：事件应描述在叙事中发生的具体动作、行为或变化，并涉及参与方、地点、时间等要素，且能推动情节发展或产生影响。纯时间节点、氛围或状态性描写（如“新的一天开始了”“天色渐暗”）不应作为事件抽取，除非伴随具体行为或情节推进。
12. 禁止把“混剪/旁白/字幕/镜头/转场”等叙事手法当作事件或实体；此类信息只可作为 evidence/note 的来源说明。使用“动作 + 对象/结果（+可选地点/时间提示）”命名，例如「领航员空间站改变航向驶向月球」；不要使用「混剪开始」「旁白说明…」等元叙事词。
13. 实体命名规范：**禁止将编号或括注（如“[73]”“（43）”“（父亲）”“（青年）”等）并入实体名称，均应去除括号内容并统一为核心名称（如“周喆直[73]”“王强（父亲）”“刘洋（青年）”→“周喆直”“王强”“刘洋”）
"""

# general_rules = """
# 1. Entity Type Identification: Do not treat category names such as Concept, Event, Object, Action as entities themselves; personal pronouns (such as "you", "I", "he") should not be used as entity names.
# 2. Clear Relationship Semantics: Extracted relationships must have clear subject-object distinction, clear semantics, definite direction, and logical validity.
# 3. Legal Relationship Types: Relationship type fields must strictly use the English enumeration values provided by the system, and no other forms are allowed; relationship names can be in natural language.
# 4. Relationship Entity Source: Only extract relationships between already identified entities, without introducing additional unidentified entities.
# 5. Extraction Decision: Relationships that cannot be clearly determined should be abandoned for extraction, avoiding guessing or generating out of thin air.
# 6. Core Entity Priority: Only retain entities related to narrative progression or core plot, ignoring purely background elements (such as waves, sea surface, explosions).
# 7. Auxiliary Information Processing: Auxiliary information such as title cards and visual prompts in scripts should not be treated as entities themselves, but specific content contained within them may be extracted as appropriate.
# 8. Entity Combination Identification: When entity names contain titles, positions, or modifiers (such as "Major", "Mr.", "Dr."), the core character should be identified (such as "Liu Peiqiang") rather than the complete modified phrase.
# 9. Entity Description: Each entity must provide a description field; if the original extraction is reasonable, there is no need to demand modifications during reflection.
# 10. Semantic Scope: Can only be "global" or "local". Those with clear identity, named and recurring across chapters/scenes are "global" (such as specific named characters or concepts, e.g., Tian Qiuyu, fake Tian Qiuyu); one-time, generic and context-dependent are "local" (such as "a researcher", "a passenger", "a box").
# 11. Event Extraction Principles: Events should describe specific actions, behaviors, or changes that occur in the narrative, involving participants, locations, time and other elements, and can drive plot development or produce impact. Pure time points, atmosphere or descriptive states (such as "a new day begins", "the sky gradually darkens") should not be extracted as events unless accompanied by specific behaviors or plot progression.
# 12. Prohibit treating "montage/narration/subtitles/shots/transitions" and other narrative techniques as events or entities; such information can only be used as sources for evidence/notes. Use "action + object/result (+ optional location/time hints)" for naming, such as "Navigator space station changes course toward the moon"; do not use meta-narrative terms like "montage begins", "narration explains...".
# """


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
3. "insights"字段，包含阅读心得与发现

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
2. "score"字段，表示当前抽取的得分
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