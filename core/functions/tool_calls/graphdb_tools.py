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
    description = (
        "按指定实体类型进行关键词/别名模糊检索。"
        "当 entity_type 无效或未提供时回退为 'Entity'；"
        "当 query 为空字符串时返回该类型下的全部实体（可能较多）。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "检索关键词，支持别名模糊匹配；可为空以列出该类型全部实体。", "required": True},
        {"name": "entity_type", "type": "string", "description": "目标实体类型；若无效将安全回退为 'Entity'。", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_entity_by_name")
        params_dict = json.loads(params)
        query = params_dict.get("query", "")
        entity_type = params_dict.get("entity_type", "Entity")
        available_entity_types = self.neo4j_utils.list_entity_types()
        if entity_type not in available_entity_types:
            logger.info("❗ 未找到实体类型，使用 Entity")
            entity_type = "Entity"

        results = self.neo4j_utils.search_entities_by_type(entity_type, keyword=query)
        return format_entity_results(results)

@register_tool("retrieve_entity_by_id")
class EntityRetrieverID(BaseTool):
    name = "retrieve_entity_by_id"
    description = (
        "根据实体 ID 返回实体信息。可选返回属性与关系（默认均为 False）。"
    )
    parameters = [
        {"name": "entity_id", "type": "string", "description": "实体唯一 ID。", "required": True},
        {"name": "contain_properties", "type": "bool", "description": "是否包含属性字段，默认 False。", "required": False},
        {"name": "contain_relations", "type": "bool", "description": "是否包含与其它实体的关系列表，默认 False。", "required": False},
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
    description = (
        "给定实体 ID，检索与之相连的相关实体。"
        "可按谓词、关系类型与实体类型过滤；"
        "支持返回 (实体, 关系) 的详细模式或仅返回实体。"
    )
    parameters = [
        {"name": "source_id", "type": "string", "description": "起点实体 ID。", "required": True},
        {"name": "predicate", "type": "string", "description": "关系谓词过滤（可选）。", "required": False},
        {"name": "relation_types", "type": "array", "description": "关系类型过滤（字符串数组，可选）。", "required": False},
        {"name": "entity_types", "type": "array", "description": "目标实体类型过滤（字符串数组，可选）。", "required": False},
        {"name": "limit", "type": "number", "description": "返回条数上限（可选）。", "required": False},
        {"name": "return_relations", "type": "bool", "description": "是否返回 (实体, 关系) 对而非仅实体，默认 False。", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config=None):
        self.neo4j_utils = neo4j_utils
        if embedding_config:
            self.neo4j_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_related_entities")
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
        logger.info("🔎 调用 get_relation_summary")
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
        logger.info("🔎 get_common_neighbors")
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

@register_tool("find_paths_between_nodes")
class FindPathsBetweenNodes(BaseTool):
    """
    在图中抽取两个节点之间的无向路径，并以自然语言格式返回。
    - 节点展示: name, id, labels, description
    - 关系展示: relation_name/predicate(type), confidence, description/reason
    """
    name = "find_paths_between_nodes"
    description = "在图中抽取两个节点之间的无向路径（证据链），返回自然语言描述。"
    parameters = [
        {"name": "src_id", "type": "string", "description": "起点节点的 id", "required": True},
        {"name": "dst_id", "type": "string", "description": "终点节点的 id", "required": True},
        {"name": "max_depth", "type": "integer", "description": "路径最大边数（默认 4）", "required": False},
        {"name": "limit", "type": "integer", "description": "返回路径条数上限（默认 5）", "required": False},
    ]

    def __init__(self, neo4j_utils):
        self.neo4j_utils = neo4j_utils

    def _shorten(self, text: str, max_len: int = 120) -> str:
        if not text:
            return ""
        text = text.replace("\n", " ")
        return text if len(text) <= max_len else text[:max_len] + "…"

    def _format_node(self, node: Dict[str, Any]) -> str:
        name = node.get("name") or "(未命名)"
        eid = node.get("id") or "N/A"
        labels = ",".join(node.get("labels", []))
        desc = self._shorten(node.get("description", ""))
        return f"**{name}** (id={eid}, labels=[{labels}]) — {desc}"

    def _format_rel(self, rel: Dict[str, Any]) -> str:
        rname = rel.get("relation_name") or rel.get("predicate") or rel.get("type") or "RELATED"
        conf = rel.get("confidence")
        conf_txt = f"(confidence={conf:.2f})" if conf is not None else ""
        desc = rel.get("properties", {}).get("description") or rel.get("reason") or ""
        desc_txt = f" — {self._shorten(desc)}" if desc else ""
        return f"── {rname}{conf_txt}{desc_txt} ──>"

    def _render_path(self, path: Dict[str, Any]) -> str:
        nodes = path.get("nodes", [])
        rels = path.get("relationships", [])
        parts = []
        for i, node in enumerate(nodes):
            parts.append(self._format_node(node))
            if i < len(rels):
                parts.append(self._format_rel(rels[i]))
        return "\n".join(parts)

    def call(self, params: Any, **kwargs) -> str:
        logger.info("🔎 调用 find_paths_between_nodes")
        try:
            data: Dict[str, Any] = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        src_id = data.get("src_id")
        dst_id = data.get("dst_id")
        if not src_id or not dst_id:
            return "❌ 必须提供 src_id 和 dst_id"

        max_depth = int(data.get("max_depth", 4))
        limit = int(data.get("limit", 5))

        try:
            paths = self.neo4j_utils.find_paths_between_nodes(
                src_id=src_id,
                dst_id=dst_id,
                max_depth=max_depth,
                limit=limit
            )
            if not paths:
                return f"⚠️ 在 {max_depth} 跳内，没有找到 {src_id} 与 {dst_id} 之间的路径。"

            lines = [f"找到 {len(paths)} 条路径："]
            for i, p in enumerate(paths, 1):
                lines.append(f"\n**路径 {i} (长度={p['length']})**\n{self._render_path(p)}")
            return "\n".join(lines)
        except Exception as e:
            logger.exception("find_paths_between_nodes 执行失败")
            return f"执行失败: {str(e)}"

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

    def __init__(self, neo4j_utils):
        self.neo4j_utils = neo4j_utils  # 依赖 neo4j_utils.top_k_by_centrality()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 top_k_by_centrality")
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

@register_tool("get_co_section_entities")
class GetCoSectionEntities(BaseTool):
    name = "get_co_section_entities"
    description = "输入实体id，返回与该实体同一章节/场次中的其它实体。"
    parameters = [
        {"name": "entity_id", "type": "string", "description": "实体ID", "required": True},
        {"name": "include_types", "type": "array", "description": "可选的实体类型过滤，如 ['Event','Character']", "required": False},
    ]

    def __init__(self, neo4j_utils):
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

@register_tool("query_similar_entities")
class QuerySimilarEntities(BaseTool):
    """
    基于向量索引的语义检索工具：输入自然语言文本，返回最相似的实体节点。
    内部使用 entityEmbeddingIndex（Neo4j GDS 向量索引）进行最近邻搜索，
    默认关闭 embedding 归一化（normalize=False），并在预处理时轻度清理中文标点。

    特点：
    - 支持 Top-K 控制；
    - 可按实体类型过滤（如 Character、Event 等），自动校验类型合法性；
    - 自动过滤低质量结果（score < min_score 默认阈值 0.0）；
    - 输出可选为紧凑列表或详细信息。
    """
    name = "query_similar_entities"
    description = "根据输入文本进行语义相似度检索，返回最接近的实体节点（支持Top-K和类型过滤，带安全校验）。"
    parameters = [
        {"name": "text", "type": "string", "required": True},
        {"name": "top_k", "type": "number", "required": False},
        {"name": "entity_types", "type": "array", "required": False},
        {"name": "include_meta", "type": "bool", "required": False},
    ]

    def __init__(self, neo4j_utils, embedding_config):
        self.neo4j_utils = neo4j_utils
        self.neo4j_utils.load_embedding_model(embedding_config)

        # 默认参数
        self._default_min_score = 0.0
        self._default_normalize = False
        self._default_strip = True

    # ---- 内部辅助 ----
    @staticmethod
    def _strip_zh_punct(text: str) -> str:
        if not isinstance(text, str):
            return text
        return text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "") \
                   .replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!").strip()

    @staticmethod
    def _labels_match(row_labels, wanted_types: Optional[List[str]]) -> bool:
        if not wanted_types:
            return True
        if not row_labels:
            return False
        return bool(set(map(str, row_labels)) & set(map(str, wanted_types)))

    @staticmethod
    def _fmt_compact(rows: List[dict]) -> str:
        if not rows:
            return "未找到相似实体。"
        lines = ["相似实体（紧凑显示）："]
        for r in rows:
            name = r.get("name") or "(未命名)"
            rid = r.get("id") or "UNKNOWN_ID"
            labels = r.get("labels") or []
            score = r.get("score")
            lab = "/".join(map(str, labels)) if labels else "未知类型"
            lines.append(f"- {name}  [ID: {rid}]  <{lab}>  score={score:.6f}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_verbose(rows: List[dict]) -> str:
        if not rows:
            return "未找到相似实体。"
        out = ["搜索到以下实体："]
        for r in rows:
            out.append(f"\n实体：{r.get('name') or '(未命名)'}")
            out.append(f"id: {r.get('id') or 'UNKNOWN_ID'}")
            if r.get("labels"):
                out.append(f"实体类型：{', '.join(map(str, r['labels']))}")
            if r.get("score") is not None:
                out.append(f"相似度分数：{r['score']:.6f}")
        return "\n".join(out)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 query_similar_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})

        text: str = data.get("text", "")
        if not text:
            return "text 不能为空。"

        top_k: int = int(data.get("top_k", 5) or 5)
        wanted_types: Optional[List[str]] = _as_list(data.get("entity_types"))
        include_meta: bool = _to_bool(data.get("include_meta"), False)

        # ---- 安全校验实体类型 ----
        if wanted_types:
            available_entity_types = self.neo4j_utils.list_entity_types()
            safe_types = []
            for t in wanted_types:
                if t not in available_entity_types:
                    logger.info(f"❗ 未找到实体类型 {t}，使用 Entity 代替")
                    safe_types.append("Entity")
                else:
                    safe_types.append(t)
            wanted_types = list(set(safe_types))  # 去重

        # 默认清洗中文符号
        if self._default_strip:
            text = self._strip_zh_punct(text)

        # 检索
        rows = self.neo4j_utils.query_similar_entities(
            text=text,
            top_k=top_k,
            normalize=self._default_normalize,
        ) or []

        # 阈值过滤 + 类型过滤
        filtered = [
            r for r in rows
            if r.get("score", 0.0) >= self._default_min_score
            and self._labels_match(r.get("labels"), wanted_types)
        ]
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return self._fmt_verbose(filtered) if include_meta else self._fmt_compact(filtered)

@register_tool("get_k_hop_subgraph")
class GetKHopSubgraph(BaseTool):
    """
    从一个或多个中心节点出发，抽取其 k-hop 邻居子图。
    ⚠️ 注意：k 不宜过大（建议 1–3），否则会导致结果过于庞大。
    """
    name = "get_k_hop_subgraph"
    description = (
        "输入一个或多个中心节点 ID，返回其 k-hop 邻居子图（包含节点与关系）。\n"
        "⚠️ 注意：k 不能太大，建议 1–3 跳，否则图会爆炸性增长。"
    )
    parameters = [
        {"name": "center_ids", "type": "array", "description": "中心节点 ID 列表", "required": True},
        {"name": "k", "type": "integer", "description": "邻居跳数，建议 1–3（默认 2）", "required": False},
        {"name": "limit_nodes", "type": "integer", "description": "返回的最大节点数上限（默认 200）", "required": False},
    ]

    def __init__(self, neo4j_utils):
        self.neo4j_utils = neo4j_utils

    def _shorten(self, text: str, max_len: int = 120) -> str:
        if not text:
            return ""
        text = text.replace("\n", " ")
        return text if len(text) <= max_len else text[:max_len] + "…"

    def _fmt_node(self, n: Dict[str, Any]) -> str:
        name = n.get("name") or "(未命名)"
        nid = n.get("id") or "N/A"
        labels = ",".join(n.get("labels", []))
        desc = self._shorten(n.get("description", ""))
        return f"- **{name}** [ID: {nid}, Labels: {labels}] — {desc}"

    def _fmt_rel(self, r: Dict[str, Any], node_map: Dict[str, str]) -> str:
        rtype = r.get("relation_name") or r.get("predicate") or r.get("type") or "RELATED"
        conf = r.get("confidence")
        conf_txt = f", confidence={conf:.2f}" if conf is not None else ""
        sname = node_map.get(r.get("start"), r.get("start"))
        tname = node_map.get(r.get("end"), r.get("end"))
        # 关系描述
        props = r.get("properties") or {}
        desc = props.get("description") or props.get("reason") or ""
        desc_txt = f" — {self._shorten(desc)}" if desc else ""
        return f"- {sname} ({r.get('start')}) -[{rtype}{conf_txt}]-> {tname} ({r.get('end')}){desc_txt}"

    def call(self, params: Any, **kwargs) -> str:
        logger.info("🔎 调用 get_k_hop_subgraph")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        center_ids = data.get("center_ids")
        if not center_ids:
            return "❌ 必须提供至少一个 center_id"

        k = int(data.get("k", 2))
        limit_nodes = int(data.get("limit_nodes", 200))

        try:
            subgraph = self.neo4j_utils.get_k_hop_subgraph(center_ids, k, limit_nodes)
            nodes = subgraph.get("nodes", [])
            rels = subgraph.get("relationships", [])

            if not nodes:
                return f"⚠️ 在 {k}-hop 内未找到子图。"

            node_map = {n["id"]: n.get("name") or n["id"] for n in nodes}

            lines = [
                f"抽取到 {len(nodes)} 个节点和 {len(rels)} 条关系 (中心节点: {', '.join(center_ids)}，跳数={k})",
                "",
                "节点："
            ]
            for n in nodes:
                lines.append(self._fmt_node(n))

            if rels:
                lines.append("\n关系：")
                for r in rels:
                    lines.append(self._fmt_rel(r, node_map))

            return "\n".join(lines)
        except Exception as e:
            logger.exception("get_k_hop_subgraph 执行失败")
            return f"执行失败: {str(e)}"


@register_tool("find_related_events_and_plots")
class FindRelatedEventsAndPlots(BaseTool):
    """
    给定一个节点 ID，查找与之关联的 Event 及其所属 Plot。
    ⚠️ 注意：max_depth 不宜过大（建议 2–3），否则搜索空间会爆炸。
    """
    name = "find_related_events_and_plots"
    description = (
        "输入一个节点 ID，返回与其关联的 Event 及其所属 Plot。\n"
        "搜索通过任意关系连接，最大深度可控；"
        "如果 Event 通过 HAS_EVENT 连接到 Plot，也会一并返回。\n"
        "⚠️ 注意：max_depth 建议 2–3，不要过大。"
    )
    parameters = [
        {"name": "entity_id", "type": "string", "description": "输入节点 ID", "required": True},
        {"name": "max_depth", "type": "integer", "description": "搜索最大深度（默认 3，建议 2–3）", "required": False},
    ]

    def __init__(self, neo4j_utils):
        self.neo4j_utils = neo4j_utils

    def _shorten(self, txt: str, max_len: int = 100) -> str:
        if not txt:
            return ""
        return txt.replace("\n", " ")[:max_len] + ("…" if len(txt) > max_len else "")

    def call(self, params: Any, **kwargs) -> str:
        logger.info("🔎 调用 find_related_events_and_plots")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"❌ 参数解析失败: {e}"

        entity_id = data.get("entity_id")
        if not entity_id:
            return "❌ 必须提供 entity_id"

        max_depth = int(data.get("max_depth", 3))

        try:
            results = self.neo4j_utils.find_related_events_and_plots(entity_id, max_depth)
            if not results:
                return f"⚠️ 在 {max_depth} 跳内未找到与 {entity_id} 相关的 Event。"

            lines = [f"找到 {len(results)} 个 Event 与 {entity_id} 相关 (max_depth={max_depth})："]

            for i, r in enumerate(results, 1):
                ev = r["event"]
                ev_name = ev.get("name") or "(未命名事件)"
                ev_id = ev.get("id")
                ev_desc = self._shorten(ev.get("description", ""))

                lines.append(f"\n{i}. **事件**: {ev_name} [ID: {ev_id}] — {ev_desc}")

                # 路径信息
                path_nodes = r.get("path_nodes") or []
                if path_nodes:
                    path_txt = " -> ".join(
                        [f"{n.get('name') or n.get('id')}({n.get('id')})" for n in path_nodes]
                    )
                    lines.append(f"   路径: {path_txt}")

                # Plot 信息
                plots = r.get("plots") or []
                if plots:
                    for pl in plots:
                        pl_name = pl.get("name") or "(未命名情节)"
                        pl_id = pl.get("id")
                        pl_desc = self._shorten(pl.get("description", ""))
                        lines.append(f"   所属情节: {pl_name} [ID: {pl_id}] — {pl_desc}")
                else:
                    lines.append("   (未找到关联的 Plot)")

            return "\n".join(lines)

        except Exception as e:
            logger.exception("find_related_events_and_plots 执行失败")
            return f"执行失败: {str(e)}"
