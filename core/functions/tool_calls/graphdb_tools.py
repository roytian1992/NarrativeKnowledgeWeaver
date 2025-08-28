from typing import Dict, Any, List, Optional
import json
import logging
from qwen_agent.tools.base import BaseTool, register_tool

from qwen_agent.utils.utils import logger

# =========================
# 公共格式化/工具函数
# =========================

def format_entity_results(results):
    lines = ["搜索到以下实体："]
    for entity in results:
        eid = getattr(entity, "id", None)
        name = getattr(entity, "name", "") or "(未命名)"

        etype = getattr(entity, "type", None)
        if isinstance(etype, (list, tuple, set)):
            type_text = ", ".join(map(str, etype))
        elif isinstance(etype, str):
            type_text = etype
        else:
            type_text = ""

        aliases_list = getattr(entity, "aliases", []) or []
        aliases_text = ", ".join(map(str, aliases_list)) if aliases_list else ""

        desc = getattr(entity, "description", None)
        props = getattr(entity, "properties", {}) or {}
        source_chunks_list = getattr(entity, "source_chunks", []) or []
        source_chunks_text = ", ".join(map(str, source_chunks_list)) if source_chunks_list else ""

        lines.append(f"\n实体：{name}")
        if eid:
            lines.append(f"id: {eid}")
        if type_text:
            lines.append(f"实体类型：{type_text}")
        if desc:
            lines.append(f"相关描述：{desc}")
        if aliases_text:
            lines.append(f"别名有：{aliases_text}")
        if source_chunks_text:
            lines.append(f"相关文档的chunk_id为：{source_chunks_text}")

        if isinstance(props, dict) and props:
            prop_lines = []
            for key, val in props.items():
                if val in (None, "", [], {}, ()):
                    continue
                if key == "name" and (val == name or val in aliases_list):
                    continue
                prop_lines.append(f"- {key}: {val}")
            if prop_lines:
                lines.append("相关属性如下：")
                lines.extend(prop_lines)

    return "\n".join(lines)


def _to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default

def _as_list(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",")]
    return [val]

def _fmt_entity_line(e) -> str:
    _labels = e.type if isinstance(e.type, list) else ([e.type] if e.type else [])
    etype = "/".join(_labels) if _labels else "未知类型"
    return f"- {e.name}  [ID: {e.id}]  <{etype}>"

def _fmt_relation_line(rel) -> str:
    pred = getattr(rel, "predicate", "") or "UNKNOWN_REL"
    rid = getattr(rel, "id", "")
    rn = ""
    try:
        rn = rel.properties.get("relation_name") or ""
    except Exception:
        rn = ""
    reason = ""
    try:
        reason = rel.properties.get("reason") or rel.properties.get("description") or ""
    except Exception:
        reason = ""
    return f"  ↳ {pred}{('('+rn+')' if rn else '')} [rel_id: {rid}]  {('理由: '+reason) if reason else ''}"

def _fmt_chain(ids: List[str], neo4j_utils) -> str:
    parts = []
    for _id in ids:
        node = neo4j_utils.get_entity_by_id(_id)
        if node:
            parts.append(f"{node.name}({_id})")
        else:
            parts.append(_id)
    return " -> ".join(parts)


# =========================
# 工具类
# =========================

@register_tool("retrieve_entity_by_name")
class EntityRetrieverName(BaseTool):
    name = "retrieve_entity_by_name"
    description = "在图数据库中检索指定类型的实体。支持关键词、别名的模糊匹配；若 query 为空字符串，则返回该类型下的所有实体。如果entity_type无法确定，就填写'Entity'"
    parameters = [
        {"name": "query", "type": "string", "description": "检索关键词", "required": True},
        {"name": "entity_type", "type": "string", "description": "实体类型，如果无法确定，就填写'Entity'", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_entity_by_name")
        params_dict = json.loads(params)
        query = params_dict.get("query", "")
        entity_type = params_dict.get("entity_type", "Entity")
        results = self.neo4j_utils.search_entities_by_type(entity_type, keyword=query)
        return format_entity_results(results)


@register_tool("retrieve_entity_by_id")
class EntityRetrieverID(BaseTool):
    name = "retrieve_entity_by_id"
    description = "根据实体ID检索实体，可选择是否返回属性与关系。"
    parameters = [
        {"name": "entity_id", "type": "string", "required": True},
        {"name": "contain_properties", "type": "bool", "required": False},
        {"name": "contain_relations", "type": "bool", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils
        if embedding_config:
            self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_entity_by_id")
        params_dict = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = params_dict.get("entity_id")
        contain_properties = _to_bool(params_dict.get("contain_properties"), False)
        contain_relations = _to_bool(params_dict.get("contain_relations"), False)
        return self.neo4j_utils.get_entity_info(entity_id, contain_properties, contain_relations)


@register_tool("search_related_entities")
class SearchRelatedEntities(BaseTool):
    name = "search_related_entities"
    description = "给定实体ID，检索其相关实体。"
    parameters = [
        {"name": "source_id", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils
        if embedding_config:
            self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_related_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        source_id = data.get("source_id")
        predicate = data.get("predicate") or None
        relation_types = _as_list(data.get("relation_types"))
        entity_types = _as_list(data.get("entity_types"))
        return_relations = _to_bool(data.get("return_relations"), False)

        results = self.neo4j_utils.search_related_entities(
            source_id=source_id,
            predicate=predicate,
            relation_types=relation_types,
            entity_types=entity_types,
            limit=data.get("limit"),
            return_relations=return_relations
        )
        if not results:
            return "未找到相关实体。"
        lines = []
        if return_relations:
            lines.append("检索到以下相关实体（含关系）：")
            for ent, rel in results:
                lines.append(_fmt_entity_line(ent))
                lines.append(_fmt_relation_line(rel))
        else:
            lines.append("检索到以下相关实体：")
            for ent in results:
                lines.append(_fmt_entity_line(ent))
        return "\n".join(lines)


@register_tool("get_relation_summary")
class GetRelationSummary(BaseTool):
    name = "get_relation_summary"
    description = "给定两个实体ID与关系类型，返回一段关系说明。"
    parameters = [
        {"name": "src_id", "type": "string", "required": True},
        {"name": "tgt_id", "type": "string", "required": True},
        {"name": "relation_type", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_relation_summary")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        return self.neo4j_utils.get_relation_summary(data["src_id"], data["tgt_id"], data["relation_type"])


@register_tool("get_common_neighbors")
class GetCommonNeighbors(BaseTool):
    name = "get_common_neighbors"
    description = "返回两个实体的共同邻居。"
    parameters = [
        {"name": "id1", "type": "string", "required": True},
        {"name": "id2", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_common_neighbors")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        return str(self.neo4j_utils.get_common_neighbors(id1=data["id1"], id2=data["id2"]))


@register_tool("query_similar_entities")
class QuerySimilarEntities(BaseTool):
    name = "query_similar_entities"
    description = "文本检索相似实体。"
    parameters = [
        {"name": "text", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 query_similar_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        rows = self.neo4j_utils.query_similar_entities(text=data["text"], top_k=data.get("top_k", 5))
        return str(rows)


@register_tool("find_event_chain")
class FindEventChain(BaseTool):
    name = "find_event_chain"
    description = "从起点事件出发，返回事件链。"
    parameters = [
        {"name": "entity_id", "type": "string", "required": True}
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 find_event_chain")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        chains = self.neo4j_utils.find_event_chain(entity_id=data["entity_id"])
        return str(chains)


@register_tool("check_nodes_reachable")
class CheckNodesReachable(BaseTool):
    name = "check_nodes_reachable"
    description = "判断两个节点是否可达。"
    parameters = [
        {"name": "src_id", "type": "string", "required": True},
        {"name": "dst_id", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 check_nodes_reachable")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        return str(self.neo4j_utils.check_nodes_reachable(src_id=data["src_id"], dst_id=data["dst_id"]))


@register_tool("top_k_by_centrality")
class TopKByCentrality(BaseTool):
    name = "top_k_by_centrality"
    description = "按中心度返回Top-K节点。"
    parameters = [
        {"name": "metric", "type": "string", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 top_k_by_centrality")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        return str(self.neo4j_utils.top_k_by_centrality(metric=data["metric"]))


@register_tool("get_co_section_entities")
class GetCoSectionEntities(BaseTool):
    name = "get_co_section_entities"
    description = "返回与该实体同一章节/场次中的其它实体。"
    parameters = [
        {"name": "entity_id", "type": "string", "description": "起始实体ID", "required": True},
        {"name": "include_types", "type": "array", "description": "可选的实体类型过滤，如 ['Event','Character']", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_co_section_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = data.get("entity_id")
        if not entity_id:
            raise ValueError("缺少必要参数：entity_id")

        include_types = _as_list(data.get("include_types"))
        results = self.neo4j_utils.find_co_section_entities(
            entity_id=entity_id,
            include_types=include_types,
        )

        if not results:
            scope = f"（类型过滤：{include_types}）" if include_types else ""
            return f"未在同一分节中找到其它实体{scope}。"

        return format_entity_results(results)

