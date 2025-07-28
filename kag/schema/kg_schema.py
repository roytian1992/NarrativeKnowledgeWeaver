# kag/schema/kg_schema.py

# ENTITY TYPES
ENTITY_TYPES = [
    {"type": "Character", "description": "人物角色"},
    {"type": "Event", "description": "事件或事件节点"},
    {"type": "Location", "description": "地点、场景、地理位置"},
    {"type": "Object", "description": "具体物品或道具"},
    {"type": "Concept", "description": "抽象概念、组织、身份、职能"},
]

RELATION_TYPE_GROUPS = {
    "character_relations": [
        {"type": "family_relation", "description": "家庭关系，如父子、兄妹"},
        {"type": "social_relation", "description": "社会关系，如同事、朋友"},
    ],
    "event_relations": [
        {"type": "participates_in", "description": "参与某个事件"},
        {"type": "causes", "description": "导致某个事件"},
        {"type": "happens_at", "description": "事件发生在某时间或地点"},
    ],
    "scene_relations": [
        {"type": "appears_in_scene", "description": "实体出现在某场景中"},
        {"type": "scene_follows", "description": "场景顺序，某场景接续另一场景"},
        {"type": "located_in_time", "description": "场景或事件的时间定位"},
    ],
    "generic_relations": [
        {"type": "has_property", "description": "实体具有某属性或特征"},
        {"type": "is_a", "description": "实体是某一类的实例"},
    ],
}