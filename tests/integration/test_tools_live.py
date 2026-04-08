import json
import os
from pathlib import Path

import pytest
from langchain_core.documents import Document as LCDocument

from core.utils.config import KAGConfig
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.graph_query_utils import GraphQueryUtils
from core.model_providers.openai_llm import OpenAILLM
from core.model_providers.openai_rerank import OpenAIRerankModel
from core.builder.manager.document_manager import DocumentParser

from core.functions.tool_calls.graphdb_tools import (
    EntityRetrieverName,
    EntityRetrieverID,
    SearchRelatedEntities,
    GetRelationSummary,
    GetCommonNeighbors,
    FindPathsBetweenNodes,
    TopKByCentrality,
    GetCoSectionEntities,
    SearchEpisodes,
    SearchStorylines,
    SearchSections,
    QuerySimilarEntities,
    QuerySimilarFacts,
    GetKHopSubgraph,
)
from core.functions.tool_calls.vectordb_tools import (
    VDBHierdocsSearchTool,
    VDBDocsSearchTool,
    VDBSentencesSearchTool,
    VDBGetDocsByDocumentIDsTool,
)
from core.functions.tool_calls.native_tools import (
    LookupTitlesByDocumentIDsTool,
    LookupDocumentIDsByTitleTool,
    BM25SearchDocsTool,
    SearchRelatedContentTool,
)
from core.functions.tool_calls.composite_tools import NarrativeHierarchicalSearch
from core.functions.tool_calls.composite_tools import SectionEvidenceSearch


@pytest.fixture(scope="session")
def live_ctx():
    cfg_path = os.environ.get("KAG_CONFIG", "configs/config_openai.yaml")
    cfg = KAGConfig.from_yaml(cfg_path)

    graph_store = GraphStore(cfg)
    graph = graph_store.get_graph()
    if graph.number_of_nodes() <= 0:
        pytest.fail("Local graph runtime is empty. Please build the graph before running live tool tests.")

    graph_query_utils = GraphQueryUtils(graph_store, doc_type=cfg.global_config.doc_type)
    graph_query_utils.load_embedding_model(cfg.embedding)

    doc_vs = VectorStore(cfg, "document")
    sent_vs = VectorStore(cfg, "sentence")

    llm = OpenAILLM(cfg)
    reranker = OpenAIRerankModel(cfg)
    document_parser = DocumentParser(cfg, llm)

    sample_ent = {"id": "", "name": ""}
    for node_id, data in graph.nodes(data=True):
        labels = set(graph_query_utils._node_labels(data))
        if "Entity" in labels or labels:
            sample_ent = {"id": node_id, "name": str(data.get("name", "") or "")}
            break

    sample_edge = {"src_id": "", "tgt_id": "", "relation_type": ""}
    sample_pair = {"a_name": sample_ent.get("name", ""), "b_name": ""}
    for src_id, tgt_id, _key, data in graph.edges(keys=True, data=True):
        pred = str(data.get("predicate") or data.get("relation_type") or "")
        if not sample_edge["src_id"]:
            sample_edge = {"src_id": src_id, "tgt_id": tgt_id, "relation_type": pred}
        if str(data.get("description", "") or "").strip():
            sample_pair = {
                "a_name": str(graph.nodes[src_id].get("name", "") or ""),
                "b_name": str(graph.nodes[tgt_id].get("name", "") or ""),
            }
            break

    sample_doc_id = ""
    for _node_id, data in graph.nodes(data=True):
        docs = list(data.get("source_documents") or [])
        if docs:
            sample_doc_id = str(docs[0] or "")
            break

    all_chunks_path = Path(cfg.knowledge_graph_builder.file_path) / "all_document_chunks.json"
    bm25_docs = []
    if all_chunks_path.exists():
        try:
            rows = json.loads(all_chunks_path.read_text(encoding="utf-8"))
            for x in (rows or [])[:300]:
                txt = str((x or {}).get("content", "") or "").strip()
                if not txt:
                    continue
                bm25_docs.append(
                    LCDocument(
                        page_content=txt,
                        metadata=(x or {}).get("metadata") or {},
                    )
                )
        except Exception:
            bm25_docs = []

    yield {
        "cfg": cfg,
        "graph_store": graph_store,
        "graph_query_utils": graph_query_utils,
        "doc_vs": doc_vs,
        "sent_vs": sent_vs,
        "llm": llm,
        "reranker": reranker,
        "document_parser": document_parser,
        "sample_ent": sample_ent,
        "sample_edge": sample_edge,
        "sample_pair": sample_pair,
        "sample_doc_id": sample_doc_id,
        "bm25_docs": bm25_docs,
    }

    graph_store.close()


def _call(tool, payload):
    out = tool.call(json.dumps(payload, ensure_ascii=False))
    assert isinstance(out, str)
    assert out.strip() != ""
    return out


def test_live_backend_health(live_ctx):
    graph_store = live_ctx["graph_store"]
    stats = graph_store.get_stats()
    assert int(stats.get("entities", 0) or 0) > 0

    doc_stats = live_ctx["doc_vs"].get_stats()
    assert doc_stats.get("status") == "connected"


@pytest.mark.parametrize("tool_cls", [SearchEpisodes, SearchStorylines, SearchSections])
def test_graph_global_search_tools(tool_cls, live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]

    if tool_cls is SearchSections:
        tool = tool_cls(graph_query_utils, embedding_config=cfg.embedding, doc_type=cfg.global_config.doc_type)
    else:
        tool = tool_cls(graph_query_utils, embedding_config=cfg.embedding)

    out = _call(tool, {"query": live_ctx["sample_ent"].get("name") or "剧情", "include_meta": True})
    assert "检索结果" in out or "未找到" in out


def test_graph_llm_filtered_global_search_tools(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]
    parser = live_ctx["document_parser"]

    storyline_tool = SearchStorylines(
        graph_query_utils,
        embedding_config=cfg.embedding,
        document_parser=parser,
        max_workers=2,
    )
    out1 = _call(
        storyline_tool,
        {
            "query": "新学员在基地开始训练和日常生活",
            "top_k": 3,
            "use_llm_filter": True,
            "include_meta": True,
        },
    )
    assert "LLM相关性" in out1 or "未找到" in out1

    episode_tool = SearchEpisodes(
        graph_query_utils,
        embedding_config=cfg.embedding,
        document_parser=parser,
        max_workers=2,
    )
    out2 = _call(
        episode_tool,
        {
            "query": "刘培强给韩朵朵送花表白",
            "top_k": 3,
            "use_llm_filter": True,
            "include_meta": True,
        },
    )
    assert "LLM相关性" in out2 or "未找到" in out2

    section_tool = SearchSections(
        graph_query_utils,
        embedding_config=cfg.embedding,
        doc_type=cfg.global_config.doc_type,
        document_parser=parser,
        max_workers=2,
    )
    out3 = _call(
        section_tool,
        {
            "query": "刘培强向韩朵朵送花表白",
            "top_k": 2,
            "use_llm_filter": True,
            "include_meta": True,
            "include_related_entities": True,
            "related_entity_limit": 2,
        },
    )
    assert "相关实体" in out3 or "未找到" in out3


def test_entity_retrieve_and_related(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]
    ent_id = live_ctx["sample_ent"].get("id")
    ent_name = live_ctx["sample_ent"].get("name") or ""
    assert ent_id

    t_name = EntityRetrieverName(graph_query_utils, cfg.embedding)
    out1 = _call(t_name, {"query": ent_name, "entity_type": "Entity"})
    assert "实体" in out1

    t_id = EntityRetrieverID(graph_query_utils)
    out2 = _call(t_id, {"entity_id": ent_id, "contain_properties": True, "contain_relations": False})
    assert ent_id in out2

    t_related = SearchRelatedEntities(graph_query_utils)
    out3 = _call(t_related, {"source_id": ent_id, "limit": 5, "return_relations": True})
    assert "相关实体" in out3 or "未找到" in out3


def test_relation_summary_neighbors_paths_subgraph(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    edge = live_ctx["sample_edge"]
    src_id = edge.get("src_id")
    tgt_id = edge.get("tgt_id")
    rel_type = edge.get("relation_type")
    if not (src_id and tgt_id and rel_type):
        pytest.skip("No sample relation in the local graph runtime.")

    out1 = _call(GetRelationSummary(graph_query_utils), {"src_id": src_id, "tgt_id": tgt_id, "relation_type": rel_type})
    assert "未找到" in out1 or len(out1) > 0

    out2 = _call(GetCommonNeighbors(graph_query_utils), {"id1": src_id, "id2": tgt_id, "include_rel_types": True, "limit": 5})
    assert "共同邻居" in out2 or "无共同邻居" in out2

    out3 = _call(FindPathsBetweenNodes(graph_query_utils), {"src_id": src_id, "dst_id": tgt_id, "max_depth": 3, "limit": 2})
    assert "路径" in out3 or "没有找到" in out3

    out4 = _call(GetKHopSubgraph(graph_query_utils), {"center_ids": [src_id], "k": 1, "limit_nodes": 30})
    assert "子图" in out4 or "未找到" in out4 or "抽取到" in out4


def test_centrality_and_co_section(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    ent_id = live_ctx["sample_ent"].get("id")

    out1 = _call(TopKByCentrality(graph_query_utils), {"metric": "degree", "top_k": 5})
    assert "Top-" in out1 or "未发现" in out1

    out2 = _call(GetCoSectionEntities(graph_query_utils), {"entity_id": ent_id})
    assert "未在同一分节" in out2 or "实体" in out2


def test_similar_entities_facts_and_dialogues(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]
    q = live_ctx["sample_ent"].get("name") or "剧情"

    out1 = _call(QuerySimilarEntities(graph_query_utils, cfg.embedding), {"text": q, "top_k": 3, "include_meta": True})
    assert "未找到" in out1 or "id:" in out1 or "相似度分数" in out1

    out2 = _call(QuerySimilarFacts(graph_query_utils, cfg.embedding), {"text": q, "top_k": 5})
    if "相关事实关系" in out2:
        assert "description:" in out2


def test_vectordb_tools(live_ctx):
    doc_vs = live_ctx["doc_vs"]
    sent_vs = live_ctx["sent_vs"]
    reranker = live_ctx["reranker"]

    q = live_ctx["sample_ent"].get("name") or "地球"

    out1 = _call(VDBDocsSearchTool(doc_vs), {"query": q, "limit": 3})
    assert "相关结果" in out1

    out2 = _call(VDBSentencesSearchTool(sent_vs), {"query": q, "limit": 3})
    assert "相关结果" in out2

    out3 = _call(VDBHierdocsSearchTool(doc_vs, sent_vs, reranker), {"query": q, "limit": 3})
    assert "相关结果" in out3

    did = live_ctx["sample_doc_id"]
    if did:
        out4 = _call(VDBGetDocsByDocumentIDsTool(doc_vs), {"document_ids": [did]})
        assert "相关结果" in out4


def test_native_doc_mapping_and_bm25_and_related_content(live_ctx):
    cfg = live_ctx["cfg"]
    doc_vs = live_ctx["doc_vs"]
    parser = live_ctx["document_parser"]

    base = Path(cfg.knowledge_graph_builder.file_path)
    index_path = str(base / "doc2chunks_index.json")
    doc2chunks_path = str(base / "doc2chunks.json")
    did = live_ctx["sample_doc_id"]
    sample_title = ""
    if Path(doc2chunks_path).exists() and did:
        try:
            rows = json.loads(Path(doc2chunks_path).read_text(encoding="utf-8"))
            md = ((rows or {}).get(did) or {}).get("document_metadata") or {}
            sample_title = str(md.get("title") or "").strip()
        except Exception:
            sample_title = ""

    if did and Path(index_path).exists():
        t_map = LookupTitlesByDocumentIDsTool(
            index_path=index_path,
            doc2chunks_path=doc2chunks_path,
            doc_type=cfg.global_config.doc_type,
        )
        out1 = _call(t_map, {"document_ids": [did]})
        assert "document_id" in out1

        t_map2 = LookupDocumentIDsByTitleTool(
            index_path=index_path,
            doc2chunks_path=doc2chunks_path,
            doc_type=cfg.global_config.doc_type,
        )
        out1b = _call(t_map2, {"title": sample_title or "数字生命研究室", "fuzzy": True, "limit": 3})
        assert "未找到" in out1b or "document_ids" in out1b

    docs = live_ctx["bm25_docs"]
    if docs:
        t_bm25 = BM25SearchDocsTool(docs)
        out2 = _call(t_bm25, {"query": (live_ctx["sample_ent"].get("name") or "地球"), "k": 3})
        assert "检索到以下文档" in out2

    if did:
        t_related = SearchRelatedContentTool(doc_vs, parser, max_workers=4)
        out3 = _call(t_related, {"document_ids": [did], "query": (live_ctx["sample_ent"].get("name") or "剧情"), "max_length": 120})
        assert out3 == "（无）" or ":" in out3


def test_narrative_hierarchical_search(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]
    doc_vs = live_ctx["doc_vs"]
    parser = live_ctx["document_parser"]

    tool = NarrativeHierarchicalSearch(
        graph_query_utils,
        doc_vs,
        parser,
        embedding_config=cfg.embedding,
        max_workers=2,
    )
    out = _call(
        tool,
        {
            "query": "刘培强和韩朵朵的感情发展到送花和等待承诺",
            "storyline_top_k": 2,
            "episode_top_k": 3,
            "event_top_k": 4,
            "document_top_k": 3,
        },
    )
    assert "[Storylines]" in out
    assert "[Episodes]" in out
    assert "[Evidence]" in out


def test_section_evidence_search(live_ctx):
    graph_query_utils = live_ctx["graph_query_utils"]
    cfg = live_ctx["cfg"]
    doc_vs = live_ctx["doc_vs"]
    parser = live_ctx["document_parser"]

    tool = SectionEvidenceSearch(
        graph_query_utils,
        doc_vs,
        parser,
        doc_type=cfg.global_config.doc_type,
        embedding_config=cfg.embedding,
        max_workers=2,
    )
    out = _call(
        tool,
        {
            "query": "刘培强向韩朵朵送花并表达心意",
            "section_top_k": 2,
            "use_llm_filter": True,
            "related_entity_limit": 2,
            "max_length": 180,
        },
    )
    assert "[Matched" in out
    assert "[Evidence]" in out
