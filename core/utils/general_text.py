# general_rules = """
# 1. 实体类型识别时，不应将 Concept、Event、Object、Action 等类别名称、人称代词（如“我”“你”“他”）以及“母亲”“父亲”等缺乏明确指向的称谓作为实体名称，只有在具备清晰指代（如“张三的母亲”）时才可作为实体。
# 2. 关系语义明确：抽取的关系需主客分明、语义清晰、方向明确，逻辑成立。
# 3. 关系类型合法：关系类型字段必须严格使用系统提供的英文枚举值，禁止使用其他形式；关系名称可为自然语言。
# 4. 关系实体来源：仅在已识别的实体之间抽取关系，不引入额外未识别的实体。
# 5. 抽取决策：无法明确判断的关系应放弃抽取，避免猜测或凭空生成。
# 6. 核心实体优先：仅保留与叙事推进或核心情节相关的实体，忽略纯背景性元素（如浪花、海面、爆炸）。
# 7. 辅助信息处理：剧本中的字卡、视觉提示等辅助信息本身不作为实体，但其中包含的具体内容可视情况抽取。
# 8. 实体组合识别：当实体名称包含称谓、职称或修饰语（如“少校”、“先生”、“博士”），应识别其核心人物（如“刘培强”），而非完整修饰短语。
# 9. 实体描述：每个实体必须提供 description 字段；若原有抽取合理，反思时不必苛求修改。
# 10. 语义范围（scope）：仅可为 "global" 或 "local"。具备明确身份、具名且跨章节/场景复现的为 "global"（如具体的出现名字的人物、概念，例如：田秋雨、假田秋雨）；一次性、泛指且依赖上下文的为 "local"（如“一名研究员”、“一位乘客”、“一个箱子”）。
# 11. 事件抽取原则：事件应描述在叙事中发生的具体动作、行为或变化，并涉及参与方、地点、时间等要素，且能推动情节发展或产生影响。纯时间节点、氛围或状态性描写（如“新的一天开始了”“天色渐暗”）不应作为事件抽取，除非伴随具体行为或情节推进。
# 12. 禁止把“混剪/旁白/字幕/镜头/转场”等叙事手法当作事件或实体；此类信息只可作为 evidence/note 的来源说明。使用“动作 + 对象/结果（+可选地点/时间提示）”命名，例如「领航员空间站改变航向驶向月球」；不要使用「混剪开始」「旁白说明…」等元叙事词。
# 13. 实体命名规范：**禁止将编号或括注（如“[73]”“（43）”“（父亲）”“（青年）”等）并入实体名称，均应去除括号内容并统一为核心名称（如“周喆直[73]”“王强（父亲）”“刘洋（青年）”→“周喆直”“王强”“刘洋”）
# """

general_rules = """
1. Entity Validity: Only extract concrete, narrative-relevant entities; schema labels (e.g., Character, Event, Object, Concept, Location, TimePoint) and personal pronouns MUST NOT be treated as entities or used as entity names.
2. Entity Naming: Normalize entity names by removing titles or modifiers (e.g., "Major", "Mr.", "Dr."), keeping only the core referent.
3. Narrative Priority: Ignore purely background, atmospheric, or auxiliary elements unless they play a direct causal or plot-driving role.
4. Event Definition: Events must represent specific actions, decisions, or state changes that advance the narrative; pure descriptions or time markers are not events.
5. Narrative Technique Exclusion: Montage, narration, subtitles, shots, transitions, and similar cinematic devices MUST NOT be extracted as entities or events.
6. Relationship Constraints: Extract only relationships with clear subject-object roles, definite direction, and logical validity.
7. Relationship Legality: Relationship types MUST strictly use the predefined English enumeration values; relationship names may be natural language.
8. Entity Reference Integrity: Only extract relationships between already identified entities; do not introduce new entities during relation extraction.
9. Conservative Policy: If an entity, event, or relationship cannot be determined with sufficient clarity, it MUST be omitted rather than inferred or fabricated.
10. Type-as-Entity Prohibition: Schema types MUST NOT appear as entity names or as targets of `is_a`; all `is_a` relations must point to a concrete Concept entity.
11. Role Disambiguation: If an entity is referred to only by a role/title (e.g., chair, guard, driver, clerk), the extracted name MUST include a grounding qualifier from the text (e.g., "meeting chair", "AA meeting chair", "security guard at the checkpoint"). Do NOT output the bare role word alone unless it is a unique proper name. Prefer the most specific in-text qualifier available.
"""


general_repair_template = """
The following response has JSON formatting issues.

Original response:
{original_response}

Error message:
{error_message}

Your task:
- Fix ONLY the JSON formatting issues.
- Do NOT change, add, remove, or reinterpret any content.
- Do NOT summarize, rewrite, or paraphrase any text.
- Preserve all original field values exactly as they are, except where required to make the JSON valid.

STRICT JSON REQUIREMENTS:
- The output MUST be valid JSON parseable by standard JSON parsers (e.g., json.loads).
- Every string value MUST be enclosed in double quotes.
- Do NOT include any unescaped double quote (") inside a string value.
  - If necessary, escape quotes as \".
- Do NOT remove required quotes around string values.
- Do NOT introduce markdown, code fences, comments, or explanations.

Return ONLY the fully corrected JSON.
"""
