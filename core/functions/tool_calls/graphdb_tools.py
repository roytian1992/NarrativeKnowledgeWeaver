from typing import Dict, Any, List, Optional
import json
from qwen_agent.tools.base import BaseTool, register_tool

# =========================
# 公共格式化/工具函数
# =========================

def format_entity_results(results):
    """
    将实体检索结果整合为自然语言文本。
    """
    text = "搜索到以下实体：\n"
    for result in results:
        id_ = result.id
        name = result.name
        entity_type = "/".join(result.type) if result.type else "未知类型"
        properties = result.properties or {}
        description = result.description or "无描述"
        
        # 拼接属性
        prop_text = ""
        for prop, value in properties.items():
            if value:  # 只输出非空
                prop_text += f"  - {prop}: {value}\n"
        
        text += f"\n实体名称：{name}\n"
        text += f"ID：{id_}\n"
        text += f"类型：{entity_type}\n"
        text += f"描述：{description}\n"
        if prop_text:
            text += f"属性：\n{prop_text}"
    return text


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
# 你的两个示例工具（保留原接口）
# =========================

@register_tool("retrieve_entity_by_name")
class EntityRetrieverName(BaseTool):
    name = "retrieve_entity_by_name"
    description = (
        "在图数据库中检索指定类型的实体。"
        "支持关键词、别名的模糊匹配；"
        "若 query 为空字符串，则返回该类型下的所有实体。"
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": (
                "检索关键词，用于匹配实体名称、别名或拼音片段。"
                "当输入为空字符串时，返回该类型的所有实体。"
            ),
            "required": True
        },
        {
            "name": "entity_type",
            "type": "string",
            "description": "实体类型（如 Character、Event、Location、Object、Concept 等）。",
            "required": True
        }
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        self.neo4j_utils.load_embedding_model(embedding_config)
    
    def call(self, params: str, **kwargs) -> str:
        # print("[CHECK] tool called!")
        params_dict = json.loads(params)
        query = params_dict.get("query", "")
        entity_type = params_dict.get("entity_type")
        results = self.neo4j_utils.search_entities_by_type(entity_type, keyword=query)
        return format_entity_results(results)


@register_tool("retrieve_entity_by_id")
class EntityRetrieverID(BaseTool):
    name = "retrieve_entity_by_id"
    description = (
        "根据实体的唯一 ID，从图数据库中精确检索该实体；"
        "可选择是否一并返回实体属性与关系。"
    )
    parameters = [
        {
            "name": "entity_id",
            "type": "string",
            "description": "目标实体的唯一标识符（ID），必须提供。",
            "required": True
        },
        {
            "name": "contain_properties",
            "type": "bool",
            "description": "是否返回实体属性（默认 False）。",
            "required": False
        },
        {
            "name": "contain_relations",
            "type": "bool",
            "description": "是否返回与该实体相关的关系（默认 False）。",
            "required": False
        }
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils
        if embedding_config:
            self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        # print("[CHECK] retrieve_entity_by_id called")
        params_dict = json.loads(params) if isinstance(params, str) else dict(params or {})

        entity_id = params_dict.get("entity_id")
        contain_properties = _to_bool(params_dict.get("contain_properties"), default=False)
        contain_relations = _to_bool(params_dict.get("contain_relations"), default=False)

        if not entity_id:
            raise ValueError("Missing required parameter: entity_id")

        # 直接返回结果（自然语言上下文）
        return self.neo4j_utils.get_entity_info(
            entity_id,
            contain_properties=contain_properties,
            contain_relations=contain_relations
        )


# =========================
# 我封装的 6 个查询类工具
# =========================

@register_tool("search_related_entities")
class SearchRelatedEntities(BaseTool):
    name = "search_related_entities"
    description = (
        "给定实体ID，检索其相关实体；可按谓词、关系类型、目标实体类型过滤。"
        "支持返回仅实体列表或附带关系信息。"
    )
    parameters = [
        {"name": "source_id", "type": "string", "description": "源实体ID（必填）", "required": True},
        {"name": "predicate", "type": "string", "description": "关系谓词过滤（如 'happens_at'）", "required": False},
        {"name": "relation_types", "type": "array", "description": "关系类型白名单（如 ['EVENT_CAUSES']）", "required": False},
        {"name": "entity_types", "type": "array", "description": "目标实体类型过滤（如 ['Event','Character']）", "required": False},
        {"name": "limit", "type": "number", "description": "返回数量上限", "required": False},
        {"name": "return_relations", "type": "bool", "description": "是否附带关系信息（默认 False）", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils
        if embedding_config:
            self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        source_id = data.get("source_id")
        if not source_id:
            raise ValueError("Missing required parameter: source_id")

        predicate = data.get("predicate") or None
        relation_types = _as_list(data.get("relation_types"))
        entity_types = _as_list(data.get("entity_types"))
        limit = data.get("limit")
        return_relations = _to_bool(data.get("return_relations"), default=False)

        results = self.neo4j_utils.search_related_entities(
            source_id=source_id,
            predicate=predicate,
            relation_types=relation_types,
            entity_types=entity_types,
            limit=int(limit) if isinstance(limit, (int, float, str)) and str(limit).isdigit() else None,
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
    description = (
        "给定两个实体ID与关系类型，返回一段可读的关系说明。"
        "若不存在该关系，返回提示。"
    )
    parameters = [
        {"name": "src_id", "type": "string", "description": "源实体ID（必填）", "required": True},
        {"name": "tgt_id", "type": "string", "description": "目标实体ID（必填）", "required": True},
        {"name": "relation_type", "type": "string", "description": "关系类型（如 'EVENT_CAUSES'）", "required": True},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        src_id = data.get("src_id")
        tgt_id = data.get("tgt_id")
        relation_type = data.get("relation_type")
        if not (src_id and tgt_id and relation_type):
            raise ValueError("缺少必要参数：src_id / tgt_id / relation_type")

        txt = self.neo4j_utils.get_relation_summary(src_id, tgt_id, relation_type)
        return txt or "未找到指定关系。"


@register_tool("get_common_neighbors")
class GetCommonNeighbors(BaseTool):
    name = "get_common_neighbors"
    description = (
        "返回两个实体的共同邻居。支持限定关系类型与方向；"
        "可选择是否附带从A/B到该邻居的关系类型列表。"
    )
    parameters = [
        {"name": "id1", "type": "string", "description": "第一个实体ID（必填）", "required": True},
        {"name": "id2", "type": "string", "description": "第二个实体ID（必填）", "required": True},
        {"name": "rel_types", "type": "array", "description": "关系类型白名单（如 ['RELATED_TO']）", "required": False},
        {"name": "direction", "type": "string", "description": "方向：any/out/in（默认 any）", "required": False},
        {"name": "limit", "type": "number", "description": "返回上限", "required": False},
        {"name": "include_rel_types", "type": "bool", "description": "是否附带从A/B出发的关系类型（默认 False）", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        id1 = data.get("id1")
        id2 = data.get("id2")
        if not (id1 and id2):
            raise ValueError("缺少必要参数：id1 / id2")

        rel_types = _as_list(data.get("rel_types"))
        direction = (data.get("direction") or "any").lower()
        if direction not in {"any", "out", "in"}:
            direction = "any"
        limit_raw = data.get("limit")
        limit = int(limit_raw) if isinstance(limit_raw, (int, float, str)) and str(limit_raw).isdigit() else None
        include_rel_types = _to_bool(data.get("include_rel_types"), default=False)

        if include_rel_types:
            items = self.neo4j_utils.get_common_neighbors_with_rels(
                id1=id1, id2=id2, rel_types=rel_types, direction=direction, limit=limit
            )
            if not items:
                return "无共同邻居。"
            lines = ["共同邻居（含从A/B的边类型）:"]
            for it in items:
                ent = it["entity"]
                lines.append(_fmt_entity_line(ent))
                lines.append(f"  ←A: {', '.join(it.get('rels_from_a', []) or [])}")
                lines.append(f"  ←B: {', '.join(it.get('rels_from_b', []) or [])}")
            return "\n".join(lines)
        else:
            ents = self.neo4j_utils.get_common_neighbors(
                id1=id1, id2=id2, rel_types=rel_types, direction=direction, limit=limit
            )
            if not ents:
                return "无共同邻居。"
            lines = ["共同邻居："]
            for e in ents:
                lines.append(_fmt_entity_line(e))
            return "\n".join(lines)


@register_tool("query_similar_entities")
class QuerySimilarEntities(BaseTool):
    name = "query_similar_entities"
    description = (
        "文本检索相似实体（向量索引）；返回 topK 近邻及相似度分数。"
    )
    parameters = [
        {"name": "text", "type": "string", "description": "查询文本（必填）", "required": True},
        {"name": "top_k", "type": "number", "description": "返回TopK（默认5）", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        # 该工具依赖 embedding 模型
        self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        text = data.get("text")
        if not text:
            raise ValueError("Missing required parameter: text")

        top_k_raw = data.get("top_k")
        top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) and str(top_k_raw).isdigit() else 5

        rows = self.neo4j_utils.query_similar_entities(text=text, top_k=top_k)
        if not rows:
            return "未检索到相似实体。"

        lines = [f"最相似实体（Top {top_k}）:"]
        for r in rows:
            name = r.get("name", "")
            _labels = r.get("labels", []) or []
            etype = "/".join(_labels) if _labels else "未知类型"
            _id = r.get("id", "")
            score = r.get("score", 0.0)
            lines.append(f"- {name}  [ID: {_id}]  <{etype}>   score={score:.4f}")
        return "\n".join(lines)


@register_tool("find_event_chain")
class FindEventChain(BaseTool):
    name = "find_event_chain"
    description = (
        "从起点事件出发，返回所有至“无出边终点”的事件链；可按边的confidence阈值过滤。"
    )
    parameters = [
        {"name": "entity_id", "type": "string", "description": "起点事件ID（必填）", "required": True}
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = data.get("entity_id")
        if not entity_id:
            raise ValueError("Missing required parameter: entity_id")


        chains = self.neo4j_utils.find_event_chain(entity_id=entity_id, min_confidence=0.0)
        if not chains:
            return "未找到符合条件的事件链。"

        lines = [f"共发现 {len(chains)} 条事件链："]
        for idx, ids in enumerate(chains, 1):
            lines.append(f"{idx}. {_fmt_chain(ids, self.neo4j_utils)}")
        return "\n".join(lines)


@register_tool("check_nodes_reachable")
class CheckNodesReachable(BaseTool):
    name = "check_nodes_reachable"
    description = (
        "判断两个节点是否可达（最短路径步数不超过 max_depth），可排除特定关系类型。"
    )
    parameters = [
        {"name": "src_id", "type": "string", "description": "起点节点ID（必填）", "required": True},
        {"name": "dst_id", "type": "string", "description": "终点节点ID（必填）", "required": True},
        {"name": "max_depth", "type": "number", "description": "最大允许路径长度（默认3）", "required": False},
        {"name": "excluded_rels", "type": "array", "description": "需要排除的关系类型列表（如 ['SCENE_CONTAINS']）", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        src_id = data.get("src_id")
        dst_id = data.get("dst_id")
        if not (src_id and dst_id):
            raise ValueError("缺少必要参数：src_id / dst_id")

        md = data.get("max_depth")
        max_depth = int(md) if isinstance(md, (int, float, str)) and str(md).isdigit() else 3
        excluded_rels = _as_list(data.get("excluded_rels"))

        ok = self.neo4j_utils.check_nodes_reachable(
            src_id=src_id, dst_id=dst_id, max_depth=max_depth, excluded_rels=excluded_rels
        )
        return f"可达性：{'是' if ok else '否'}（max_depth={max_depth}，excluded_rels={excluded_rels or []}）"

@register_tool("top_k_by_centrality")
class TopKByCentrality(BaseTool):
    name = "top_k_by_centrality"
    description = (
        "按中心度指标返回 Top-K 节点排名（已写回到节点属性的中心度）。"
        "支持的指标：pagerank/pr、degree/deg、betweenness/btw。"
        "可选按节点标签过滤（如 ['Plot','Event']）。"
    )
    parameters = [
        {
            "name": "metric",
            "type": "string",
            "description": "中心度指标：pagerank、degree、betweenness三选一。",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "number",
            "description": "返回数量，默认 50；<=0 表示不限制（大图不建议）。",
            "required": False,
        },
        {
            "name": "node_labels",
            "type": "array",
            "description": "可选的节点标签过滤（如 ['Plot','Event']）；不传表示全图。",
            "required": False,
        },
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils  # 依赖 neo4j_utils.top_k_by_centrality()

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        metric_in = (data.get("metric") or "").strip().lower()
        metric_map = {
            "pagerank": "pagerank", "pr": "pagerank",
            "degree": "degree", "deg": "degree",
            "betweenness": "betweenness", "btw": "betweenness",
        }
        if metric_in not in metric_map:
            raise ValueError("metric 仅支持：pagerank/pr、degree/deg、betweenness/btw（不支持 closeness）")

        metric = metric_map[metric_in]
        top_k_raw = data.get("top_k", 50)
        top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) and str(top_k_raw).lstrip("-").isdigit() else 50
        node_labels = data.get("node_labels")
        if isinstance(node_labels, str):
            node_labels = [s.strip() for s in node_labels.split(",") if s.strip()]
        elif node_labels is not None and not isinstance(node_labels, list):
            node_labels = [node_labels]

        # 调用底层工具方法（内部已用 n.`prop` IS NOT NULL 语法，兼容 Neo4j 5+）
        rows: List[Dict[str, Any]] = self.neo4j_utils.top_k_by_centrality(
            metric=metric,
            top_k=top_k,
            node_labels=node_labels,
        )

        if not rows:
            scope = f"{node_labels}" if node_labels else "全图"
            return f"{scope} 未发现含有该中心度属性的节点（请先运行中心度写回过程）。"

        # 格式化输出
        header = f"Top-{top_k if top_k and top_k > 0 else 'ALL'} by {metric.upper()}" + (f" @labels={node_labels}" if node_labels else "")
        lines = [header + ":"]
        for i, r in enumerate(rows, 1):
            name = r.get("name") or "(无名)"
            nid = r.get("id") or ""
            labs = r.get("labels") or []
            score = r.get("score")
            labs_txt = "/".join(labs) if labs else "Unknown"
            score_txt = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"{i:>2}. {name}  [ID: {nid}]  <{labs_txt}>  {metric}={score_txt}")
        return "\n".join(lines)
