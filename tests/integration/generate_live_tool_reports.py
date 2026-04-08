from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from langchain_core.documents import Document as LCDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.builder.manager.document_manager import DocumentParser
from core.functions.tool_calls.graphdb_tools import (
    EntityRetrieverID,
    EntityRetrieverName,
    FindPathsBetweenNodes,
    GetCommonNeighbors,
    GetCoSectionEntities,
    GetKHopSubgraph,
    GetRelationsBetweenEntities,
    QuerySimilarEntities,
    QuerySimilarFacts,
    SearchEpisodes,
    SearchRelatedEntities,
    SearchSections,
    SearchStorylines,
    TopKByCentrality,
)
from core.functions.tool_calls.native_tools import (
    BM25SearchDocsTool,
    LookupTitlesByDocumentIDsTool,
    LookupDocumentIDsByTitleTool,
    SearchRelatedContentTool,
)
from core.functions.tool_calls.sqldb_tools import (
    SQLGetInteractionsByDocumentIDs,
    SQLSearchDialogues,
    SQLSearchInteractions,
)
from core.functions.tool_calls.vectordb_tools import (
    VDBDocsSearchTool,
    VDBGetDocsByDocumentIDsTool,
    VDBHierdocsSearchTool,
    VDBSentencesSearchTool,
)
from core.model_providers.openai_llm import OpenAILLM
from core.model_providers.openai_rerank import OpenAIRerankModel
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.config import KAGConfig
from core.utils.graph_query_utils import GraphQueryUtils


REPORT_DIR = REPO_ROOT / "reports"
OUTPUT_LIMIT = 2400


@dataclass
class CaseResult:
    tool_name: str
    status: str
    payload: Dict[str, Any]
    elapsed_ms: int
    output: str
    notes: str = ""


def _truncate(text: str, limit: int = OUTPUT_LIMIT) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n... [truncated]"


def _json_block(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _write_report(path: Path, title: str, results: List[CaseResult]) -> None:
    lines: List[str] = [f"# {title}", ""]
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status != "PASS")
    lines.append(f"- Total tools: {len(results)}")
    lines.append(f"- PASS: {passed}")
    lines.append(f"- FAIL: {failed}")
    lines.append("")

    for idx, item in enumerate(results, 1):
        lines.append(f"## {idx}. `{item.tool_name}`")
        lines.append(f"- Status: `{item.status}`")
        lines.append(f"- Runtime: `{item.elapsed_ms} ms`")
        if item.notes:
            lines.append(f"- Notes: {item.notes}")
        lines.append("- Input(payload):")
        lines.append("```json")
        lines.append(_json_block(item.payload))
        lines.append("```")
        lines.append(f"- Output length: `{len(item.output)}`")
        lines.append("- Output sample:")
        lines.append("```text")
        lines.append(_truncate(item.output))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _load_bm25_docs(all_chunks_path: Path) -> List[LCDocument]:
    if not all_chunks_path.exists():
        return []
    rows = json.loads(all_chunks_path.read_text(encoding="utf-8"))
    docs: List[LCDocument] = []
    for x in rows or []:
        txt = str((x or {}).get("content", "") or "").strip()
        if not txt:
            continue
        docs.append(
            LCDocument(
                page_content=txt,
                metadata=(x or {}).get("metadata") or {},
            )
        )
    return docs


def _find_interaction_db(sql_dir: Path) -> Path:
    candidate = sql_dir / "Interaction.db"
    if candidate.exists():
        return candidate
    for p in sorted(sql_dir.iterdir()):
        if not p.is_file():
            continue
        conn = sqlite3.connect(str(p))
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Interaction_info'")
            if cur.fetchone():
                return p
        finally:
            conn.close()
    raise FileNotFoundError("Interaction_info table not found in data/sql")


def _fetch_sql_samples(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM Interaction_info
            WHERE interaction_type='dialogue'
            ORDER BY id
            LIMIT 1
            """
        )
        dialogue = dict(cur.fetchone())

        cur.execute(
            """
            SELECT *
            FROM Interaction_info
            WHERE interaction_type<>'dialogue'
            ORDER BY id
            LIMIT 1
            """
        )
        interaction = dict(cur.fetchone())

        cur.execute(
            """
            SELECT document_id, COUNT(*) AS c
            FROM Interaction_info
            GROUP BY document_id
            ORDER BY c DESC, document_id
            LIMIT 2
            """
        )
        top_doc_ids = [str(r["document_id"]) for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "dialogue": dialogue,
        "interaction": interaction,
        "top_document_ids": top_doc_ids,
    }


def _fetch_graph_samples(neo: GraphQueryUtils) -> Dict[str, Any]:
    graph = neo.graph_store.get_graph()

    main_character: Dict[str, Any] = {}
    fallback_character: Dict[str, Any] = {}
    for node_id, data in graph.nodes(data=True):
        labels = set(neo._node_labels(data))
        if "Character" not in labels:
            continue
        row = {
            "id": node_id,
            "name": str(data.get("name", "") or ""),
            "docs": list(data.get("source_documents") or []),
        }
        if row["name"] == "刘培强":
            main_character = row
            break
        score = float(data.get("degree", 0.0) or 0.0)
        if not fallback_character or score > float(fallback_character.get("_score", 0.0)):
            row["_score"] = score
            fallback_character = row
    if not main_character:
        main_character = {k: v for k, v in fallback_character.items() if k != "_score"}

    common_pair: Dict[str, Any] = {}
    ug = graph.to_undirected()
    best_common = -1
    nodes = list(graph.nodes())
    for i, id1 in enumerate(nodes):
        for id2 in nodes[i + 1 :]:
            common = len(list(nx.common_neighbors(ug, id1, id2))) if ug.has_node(id1) and ug.has_node(id2) else 0
            if common > best_common:
                best_common = common
                common_pair = {
                    "id1": id1,
                    "name1": str(graph.nodes[id1].get("name", "") or ""),
                    "id2": id2,
                    "name2": str(graph.nodes[id2].get("name", "") or ""),
                    "c": common,
                }

    relation_edge: Dict[str, Any] = {}
    for src_id, dst_id, _key, data in graph.edges(keys=True, data=True):
        relation_edge = {
            "src_id": src_id,
            "src_name": str(graph.nodes[src_id].get("name", "") or ""),
            "dst_id": dst_id,
            "dst_name": str(graph.nodes[dst_id].get("name", "") or ""),
            "relation_type": str(data.get("predicate") or data.get("relation_type") or ""),
        }
        break

    section: Dict[str, Any] = {}
    for node_id, data in graph.nodes(data=True):
        if "Scene" not in set(neo._node_labels(data)):
            continue
        section = {
            "id": node_id,
            "name": str(data.get("name", "") or ""),
            "docs": list(data.get("source_documents") or []),
        }
        break

    return {
        "main_character": main_character,
        "common_pair": common_pair,
        "relation_edge": relation_edge,
        "section": section,
    }


def _load_doc_mapping_sample(doc2chunks_path: Path) -> Dict[str, str]:
    obj = json.loads(doc2chunks_path.read_text(encoding="utf-8"))
    for document_id, pack in obj.items():
        md = (pack or {}).get("document_metadata") or {}
        title = str(md.get("title") or "").strip()
        subtitle = str(md.get("subtitle") or "").strip()
        return {
            "document_id": str(document_id),
            "title": title,
            "subtitle": subtitle,
        }
    raise RuntimeError("doc2chunks.json is empty")


def _run_case(tool_name: str, tool_obj: Any, payload: Dict[str, Any], notes: str = "") -> CaseResult:
    started = time.time()
    try:
        output = tool_obj.call(json.dumps(payload, ensure_ascii=False))
        status = "PASS" if str(output or "").strip() else "FAIL"
        text = str(output or "")
        if not text.strip():
            text = "工具返回空字符串。"
    except Exception as exc:
        status = "FAIL"
        text = f"{type(exc).__name__}: {exc}"
    elapsed_ms = int((time.time() - started) * 1000)
    return CaseResult(
        tool_name=tool_name,
        status=status,
        payload=payload,
        elapsed_ms=elapsed_ms,
        output=text,
        notes=notes,
    )


def main() -> None:
    os.chdir(REPO_ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    graph_store = GraphStore(cfg)
    neo = GraphQueryUtils(graph_store, doc_type=cfg.global_config.doc_type)
    neo.load_embedding_model(cfg.embedding)

    doc_vs = VectorStore(cfg, "document")
    sent_vs = VectorStore(cfg, "sentence")
    llm = OpenAILLM(cfg)
    reranker = OpenAIRerankModel(cfg)
    document_parser = DocumentParser(cfg, llm)

    try:
        sql_db_path = _find_interaction_db(REPO_ROOT / cfg.storage.sql_database_path)
        sql_samples = _fetch_sql_samples(sql_db_path)
        graph_samples = _fetch_graph_samples(neo)
        mapping_sample = _load_doc_mapping_sample(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "doc2chunks.json")
        bm25_docs = _load_bm25_docs(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "all_document_chunks.json")

        main_character = graph_samples["main_character"]
        common_pair = graph_samples["common_pair"]
        relation_edge = graph_samples["relation_edge"]
        scene_doc_id = mapping_sample["document_id"]
        scene_title = mapping_sample["title"]
        dialogue = sql_samples["dialogue"]
        interaction = sql_samples["interaction"]
        interaction_doc_ids = sql_samples["top_document_ids"]

        graph_tools = [
            (
                "retrieve_entity_by_name",
                EntityRetrieverName(neo, cfg.embedding),
                {"query": main_character["name"], "entity_type": "Character"},
                "使用已存在的主角实体名称做精确/模糊检索。",
            ),
            (
                "retrieve_entity_by_id",
                EntityRetrieverID(neo),
                {"entity_id": main_character["id"], "contain_properties": True, "contain_relations": False},
                "返回实体属性，不展开全部关系，避免输出过长。",
            ),
            (
                "search_related_entities",
                SearchRelatedEntities(neo),
                {"source_id": main_character["id"], "limit": 5, "return_relations": True},
                "用图谱中的高频角色检索其相邻实体与边。",
            ),
            (
                "get_relations_between_entities",
                GetRelationsBetweenEntities(neo),
                {
                    "src_id": relation_edge["src_id"],
                    "tgt_id": relation_edge["dst_id"],
                    "relation_type": relation_edge["relation_type"],
                },
                "直接使用图中已有的一条边。",
            ),
            (
                "get_common_neighbors",
                GetCommonNeighbors(neo),
                {
                    "id1": common_pair["id1"],
                    "id2": common_pair["id2"],
                    "include_rel_types": True,
                    "limit": 5,
                },
                "使用共同邻居数最高的一对角色，尽量保证非空输出。",
            ),
            (
                "find_paths_between_nodes",
                FindPathsBetweenNodes(neo),
                {
                    "src_id": common_pair["id1"],
                    "dst_id": common_pair["id2"],
                    "max_depth": 2,
                    "limit": 2,
                },
                "仍使用同一对高连通角色，避免路径为空。",
            ),
            (
                "top_k_by_centrality",
                TopKByCentrality(neo),
                {"metric": "pagerank", "top_k": 5, "node_labels": ["Character"]},
                "直接读取已写回的中心度属性。",
            ),
            (
                "get_co_section_entities",
                GetCoSectionEntities(neo),
                {"entity_id": main_character["id"], "include_types": ["Character", "Event"]},
                "查看主角所在章节/场次中的其它实体。",
            ),
            (
                "search_episodes",
                SearchEpisodes(neo, embedding_config=cfg.embedding),
                {
                    "query": "请找出刘培强第一次接近并观察行星发动机的情节。",
                    "top_k": 3,
                    "include_meta": True,
                },
                "关键词主召回 + 向量补充，query 用自然语言描述。",
            ),
            (
                "search_storylines",
                SearchStorylines(neo, embedding_config=cfg.embedding),
                {
                    "query": "请找出太空电梯升空时，学员承受生理和心理双重考验的故事线。",
                    "top_k": 3,
                    "include_meta": True,
                },
                "query 用自然语言描述。",
            ),
            (
                "search_sections",
                SearchSections(neo, embedding_config=cfg.embedding, doc_type=cfg.global_config.doc_type),
                {
                    "query": "请找出印度数字生命研究室里演示脑机接口和550A的场次。",
                    "top_k": 3,
                    "include_meta": True,
                },
                "检索 Scene 超节点，query 用自然语言描述。",
            ),
            (
                "query_similar_entities",
                QuerySimilarEntities(neo, cfg.embedding),
                {
                    "text": "那个在试飞中接近行星发动机、第一次直观看到这座庞然巨物的人物。",
                    "top_k": 5,
                    "entity_types": ["Character"],
                    "include_meta": True,
                },
                "向量检索 query 使用自然语言人物描述。",
            ),
            (
                "retrieve_triple_facts",
                QuerySimilarFacts(neo, cfg.embedding),
                {
                    "text": "找出马兆指定图恒宇与其共同负责550C系统运行这件事相关的事实关系。",
                    "top_k": 5,
                },
                "关系向量检索 query 使用自然语言描述。",
            ),
            (
                "get_k_hop_subgraph",
                GetKHopSubgraph(neo),
                {"center_ids": [main_character["id"]], "k": 1, "limit_nodes": 20},
                "抽取主角周边 1-hop 子图。",
            ),
        ]

        sql_tools = [
            (
                "search_dialogues",
                SQLSearchDialogues(str(sql_db_path), doc_type=cfg.global_config.doc_type),
                {
                    "subject": dialogue["subject_name"],
                    "object": dialogue["object_name"],
                    "document_id": dialogue["document_id"],
                    "limit": 5,
                },
                "使用 Interaction.db 中实际存在的对白样本。",
            ),
            (
                "search_interactions",
                SQLSearchInteractions(str(sql_db_path), doc_type=cfg.global_config.doc_type),
                {
                    "subject": interaction["subject_name"],
                    "object": interaction["object_name"],
                    "polarity": interaction["polarity"],
                    "limit": 5,
                },
                "使用 Interaction.db 中实际存在的非对白交互样本。",
            ),
            (
                "get_interactions_by_document_ids",
                SQLGetInteractionsByDocumentIDs(str(sql_db_path), doc_type=cfg.global_config.doc_type),
                {
                    "document_ids": interaction_doc_ids,
                    "limit": 8,
                },
                "选取交互记录最多的 document_id 做批量查询。",
            ),
        ]

        native_tools = [
            (
                "lookup_titles_by_document_ids",
                LookupTitlesByDocumentIDsTool(
                    index_path=str(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "doc2chunks_index.json"),
                    doc2chunks_path=str(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "doc2chunks.json"),
                    doc_type=cfg.global_config.doc_type,
                ),
                {"document_ids": [scene_doc_id]},
                "批量反查真实 document_id 的标题映射。",
            ),
            (
                "lookup_document_ids_by_title",
                LookupDocumentIDsByTitleTool(
                    index_path=str(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "doc2chunks_index.json"),
                    doc2chunks_path=str(REPO_ROOT / cfg.knowledge_graph_builder.file_path / "doc2chunks.json"),
                    doc_type=cfg.global_config.doc_type,
                ),
                {"title": scene_title, "fuzzy": True, "limit": 3},
                "根据真实标题反查 document_id，支持模糊匹配。",
            ),
            (
                "search_related_content",
                SearchRelatedContentTool(doc_vs, document_parser, max_workers=4),
                {
                    "document_ids": [scene_doc_id],
                    "query": "这一场里是如何演示脑机接口和550A的？",
                    "max_length": 180,
                },
                "对单个真实 document_id 做相关内容抽取，query 用自然语言。",
            ),
            (
                "bm25_search_docs",
                BM25SearchDocsTool(bm25_docs),
                {
                    "query": "哪一段在描述印度研究室里借助550A演示脑机接口技术？",
                    "k": 3,
                },
                "BM25 使用自然语言问法，但底层仍是关键词检索。",
            ),
        ]

        vectordb_tools = [
            (
                "vdb_search_hierdocs",
                VDBHierdocsSearchTool(doc_vs, sent_vs, reranker),
                {
                    "query": "请找出太空电梯升空过程中，学员承受生理和心理考验的段落。",
                    "limit": 3,
                },
                "父子文档检索，query 用自然语言描述。",
            ),
            (
                "vdb_search_docs",
                VDBDocsSearchTool(doc_vs),
                {
                    "query": "哪一段在描写印度研究室里借助550A演示脑机接口技术？",
                    "limit": 3,
                },
                "文档级向量检索，query 用自然语言描述。",
            ),
            (
                "vdb_get_docs_by_document_ids",
                VDBGetDocsByDocumentIDsTool(doc_vs),
                {"document_ids": [scene_doc_id]},
                "直接按真实 document_id 获取所有 chunk。",
            ),
            (
                "vdb_search_sentences",
                VDBSentencesSearchTool(sent_vs),
                {
                    "query": "哪一句提到人的感知和记忆本质上都是脑电波信号？",
                    "limit": 3,
                },
                "句子级向量检索，query 用自然语言描述。",
            ),
        ]

        grouped = {
            "graphdb_tools_live_report.md": ("GraphDB Tools Live Report", graph_tools),
            "sqldb_tools_live_report.md": ("SQLDB Tools Live Report", sql_tools),
            "native_tools_live_report.md": ("Native Tools Live Report", native_tools),
            "vectordb_tools_live_report.md": ("VectorDB Tools Live Report", vectordb_tools),
        }

        for filename, (title, cases) in grouped.items():
            results = [_run_case(tool_name, tool_obj, payload, notes) for tool_name, tool_obj, payload, notes in cases]
            _write_report(REPORT_DIR / filename, title, results)

        summary = {
            "scene_doc_id": scene_doc_id,
            "scene_title": scene_title,
            "main_character": main_character,
            "common_pair": common_pair,
            "relation_edge": relation_edge,
            "interaction_db": str(sql_db_path),
        }
        (REPORT_DIR / "tool_live_report_context.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        graph_store.close()


if __name__ == "__main__":
    main()
