# 当前 Agent 系统总览

这份文档总结当前 `NarrativeKnowledgeWeaver_langgraph` 里，从知识抽取到检索问答的真实实现路径。它以现在仓库代码为准，不按旧版 `NarrativeKnowledgeWeaver` 或历史 Neo4j 设计来写。

当前系统的核心特点是：

- `JSON-first`
- 本地 `NetworkX / GraphStore` 运行时图
- `Chroma` 文档级和句子级向量库
- `SQLite` interactions 数据库
- 一个以工具调用为核心的 retrieval agent

当前默认 QA backend 仍是 `langchain` 封装，但外部统一通过 `QuestionAnsweringAgent` 门面切换，支持：

- `langchain`
- `qwen`
- `openai_agents`

对应入口在 [retriever_agent.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent.py)。

## 1. 总体架构

当前主链路可以理解成 5 层：

1. 文本切块与基础抽取
2. 基础图谱与文档节点构建
3. narrative 聚合与辅助索引构建
4. 检索工具装配
5. retrieval-first QA

QUALITY 等评测实验使用的是“按文章独立 workspace”的运行方式。每篇文章会有自己的：

- `knowledge_graph/`
- `narrative_graph/`
- `vector_store/`
- `sql/`
- `interactions/`

这套结构在 [run_quality_benchmark.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/run_quality_benchmark.py) 里由 `_build_article_config()` 统一重定向。

## 2. 知识抽取流水线

基础流水线由 [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py) 驱动。

### 2.1 切块与索引中间产物

`prepare_chunks()` 负责：

- 从输入 JSON 加载文档
- 生成稳定的 `raw_doc_id / doc_segment_id / document_id / chunk_id`
- 产出 `all_document_chunks.json`
- 产出 `doc2chunks.json`
- 产出 `doc2chunks_index.json`
- 可选把 document/sentence chunk 写入向量库

这里已经不是“先入 Neo4j 再查”的模式，而是先把 JSON 产物和本地索引建好。

### 2.2 实体关系抽取

实体关系抽取由：

- [knowledge_extraction_agent.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/knowledge_extraction_agent.py)
- [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py)

共同完成。

主流程是：

1. 对 document / chunk 进行 entity-relation extraction
2. 保存原始 extraction 结果
3. 做 refinement / merge
4. 生成 `entity_basic_info.json` 与 `relation_basic_info.json`
5. 再进一步生成 refined 版本

### 2.3 图后处理

`postprocess_and_save()` 会把 refined 实体和关系写成：

- `relation_info_refined.json`
- `graph_nx.pkl`

这里已经明确是 `nx.MultiDiGraph()`，不依赖 Neo4j。

### 2.4 属性抽取

`extract_properties()` 在 refined 图基础上补实体属性，产出：

- `entity_info_refined.json`

这一步依赖图结构和 entity schema，不是简单从文本直接补字段。

### 2.5 interaction 抽取与 SQL 持久化

interaction 相关逻辑由：

- [interaction_extraction_agent.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/interaction_extraction_agent.py)
- [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py)

负责。

流程是：

1. 在 refined entity 基础上筛角色和物体候选
2. 按文档抽取 interaction records
3. 写出：
   - `interactions/interaction_results.json`
   - `interactions/interaction_records_list.json`
4. 再通过 `store_interactions_to_sql()` 写入：
   - `sql/Interaction.db`

当前 SQL 是 interaction 的存储层，不是主图数据库。

## 3. 文档节点与本地图运行时图

为了让 QA 能检索 section / scene / chapter 级节点，系统会显式构建文档超级节点。

相关逻辑在 [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py)：

- `build_doc_entities()`
- `load_json_to_graph_store()`

`build_doc_entities()` 会产出：

- `doc_entities.json`
- `doc_entity_edges.json`

其中边通常是 `CONTAINS` 风格，把 section/document 节点和普通实体连起来。

`load_json_to_graph_store()` 会把以下 JSON 重新加载到本地图运行时图：

- refined entities
- refined relations
- doc entities
- doc-entity edges

并最终保存到：

- `knowledge_graph/graph_runtime_langgraph.pkl`

这就是现在 QA 运行时真正依赖的图基础。

## 4. Narrative Graph 聚合

narrative 聚合由 [narrative_graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/narrative_graph_builder.py) 负责。

它的目标是把基础事件图提升成叙事层级：

- `Event`
- `Episode`
- `Storyline`

主阶段包括：

1. `extract_episodes()`
2. 构建 episode support edges
3. 计算 episode 之间关系
4. cycle break
5. 提取 storyline candidates
6. 构建 storyline 与 support edges / relations

典型产物在：

- `narrative_graph/episodes/`
- `narrative_graph/global/episodes.json`
- `narrative_graph/global/episode_relations.json`
- `narrative_graph/global/storylines.json`
- `narrative_graph/global/storyline_relations.json`

当前 QUALITY 主配置通常使用 `aggregation_mode: narrative`，所以 narrative 资产是 QA 主路径的一部分。

## 5. 向量库与检索索引

当前系统至少维护两套主要向量库：

- `vector_store/document`
- `vector_store/sentence`

对应逻辑见：

- [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py)
- [run_quality_benchmark.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/run_quality_benchmark.py)

关键点：

- 文档级和句子级 embedding 不是每次问答都现算
- QUALITY workspace 会用 `doc2chunks.json` 或 `all_document_chunks.json` 自动重建 vector store
- `_ensure_workspace_vector_stores_current()` 会根据 source 文件时间戳做增量判断

也就是说，现在合理模式是：

- 文章加载完成后建库
- QA 时直接检索

而不是每道题重新编码 nodes/relations。

## 6. Retriever Agent 的组织方式

统一入口是 [retriever_agent.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent.py)。

这个文件本身只是一个 facade：

- 解析 backend
- 动态加载 backend module
- 暴露统一 `QuestionAnsweringAgent`

当前默认 backend：

- `langchain`

真实实现主体在 [retriever_agent_langchain.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent_langchain.py)。

### 6.1 agent 初始化时做什么

`QuestionAnsweringAgent.__init__()` 大致会初始化：

- `GraphStore`
- `GraphQueryUtils`
- `OpenAILLM` 三套 profile
  - `retriever_llm`
  - `router_llm`
  - `memory_llm`
- `DocumentParser`
- `VectorStore(document/sentence/community)`
- `SectionTreeRetriever`
- `NarrativeTreeRetriever`
- 可选 `RetrievalStrategyMemory`
- 可选 online learning 组件
- `RetrievalToolRouter`
- 全部工具集合
- `LangGraphAssistantRuntime`

这意味着当前 agent 不是一个纯 prompt wrapper，而是有完整运行时对象图。

### 6.2 当前 QA 是 retrieval-first

默认 system message 在 [retriever_agent_langchain.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent_langchain.py) 里明确要求：

- 先检索再回答
- 不要跳过必要信息
- 多选题要比较选项和证据

`ask()` 是单轮调用，不维护多轮对话历史。

## 7. 当前 agent 可见的工具

### 7.1 base tools

`_build_base_tools()` 当前会挂载这些核心工具：

- `retrieve_entity_by_name`
- `retrieve_entity_by_id`
- `search_sections`
- `search_related_entities`
- `get_entity_sections`
- `get_relations_between_entities`
- `get_common_neighbors`
- `query_similar_facts`
- `find_paths_between_nodes`
- `top_k_by_centrality`
- `get_co_section_entities`
- `get_k_hop_subgraph`
- `vdb_search_docs`
- `vdb_get_docs_by_document_ids`
- `vdb_search_sentences`
- `vdb_search_hierdocs`
- `section_evidence_search`
- `choice_grounded_evidence_search`
- `entity_event_trace_search`
- `fact_timeline_resolution_search`
- `bm25_search_docs`
- `lookup_titles_by_document_ids`
- `lookup_document_ids_by_title`
- `search_related_content`
- 可选 SQL tools

### 7.2 aggregation tools

在 `narrative` 模式下，额外挂载：

- `narrative_hierarchical_search`

如果切到 `community` 或 `full`，还会有 community 相关工具，但当前主实验基本以 `narrative` 为主。

### 7.3 hidden tools

当前 agent 还支持 `hidden_tool_names` 配置。

这意味着：

- 某些工具可以保留实现
- 但不暴露给顶层 agent

需要注意的是，某些 composite tool 即使顶层不可见，也可以在内部继续调用 helper tool。典型例子就是：

- `search_related_content`

它既可以作为顶层工具存在，也会被 composite tools 内部复用。

## 8. 两类 tree/composite retrieval

当前比较关键的 composite 检索工具有：

- `section_evidence_search`
- `narrative_hierarchical_search`
- `choice_grounded_evidence_search`
- `entity_event_trace_search`
- `fact_timeline_resolution_search`

它们的实现集中在 [composite_tools.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/functions/tool_calls/composite_tools.py)。

### 8.1 section_evidence_search

目标：

- 找最相关的 section / scene / chapter
- 再抽相关 document 证据

它优先走 `SectionTreeRetriever`；没有时才退化到图搜索 + `search_related_content`。

输出结构通常包含：

- matched sections
- related entities
- documents
- evidence snippets

### 8.2 narrative_hierarchical_search

目标：

- 按 `Storyline -> Episode -> Event -> document evidence` 做层级叙事检索

它优先走 `NarrativeTreeRetriever`；没有时退化为图侧 `search_episode_storyline_candidates()`。

适合：

- 剧情主线
- 跨段因果
- 阶段推进
- 事件关联

### 8.3 choice_grounded_evidence_search

这是现在非常关键的 MCQ 工具。

它会：

1. 解析题目和选项
2. 对每个选项并发检索
3. 比较每个选项的支持证据
4. 输出推荐选项和证据

适合：

- `why`
- `implication`
- `attitude`
- `except`
- `least-supported`

这类必须逐选项比较的问题。

### 8.4 entity_event_trace_search

目标是把实体、事件、局部段落、narrative 证据串成“实体事件轨迹”。

它适合：

- 角色经历
- 某人为何这样做
- 某对象在剧情中的变化

### 8.5 fact_timeline_resolution_search

目标是处理 chronology / ordering / time comparison 类问题。

它适合：

- before / after
- 时间线
- 顺序判断
- 先后关系冲突消解

## 9. Tool Router

当前 router 在：

- [tool_router.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retrieval/tool_router.py)
- [tool_routing_heuristics.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retrieval/tool_routing_heuristics.py)

它不是单纯 hard-coded 的 first-tool 表，而是混合了：

- deterministic parse
- tool capability inference
- family registry
- heuristic boosts
- LLM router

当前 `RetrievalToolRouter` 会做的事情是：

1. 解析 query 类型
2. 判断是否 MCQ、是否是否定题、是否需要 chronology / narrative
3. 基于工具描述、参数和 family 生成候选工具卡片
4. 产出分阶段 tool execution plan

也就是说，router 的输出不是“直接回答”，而是“先让 agent 看哪一小组工具，再怎么升级”。

## 10. Strategy Memory

检索策略记忆在 [retrieval_strategy_memory.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/memory/retrieval_strategy_memory.py)。

当前它是 runtime-only reader，核心职责是：

- 抽象 query pattern
- 从策略库匹配模板
- 生成 runtime context
- 决定是否需要 subagent

当前主数据结构包括：

- `query_pattern`
- `candidate_patterns`
- `patterns`
- `routing_hint`

其中：

- `prepare_read_context()` 负责运行时读取
- `collect_runtime_matched_patterns()` 负责筛掉不兼容模板
- `deduplicate_patterns_for_subagents()` 负责做分支去重

注意：

- `no strategy` 不等于没有 router
- 它的含义更准确是“不读策略库”
- 是否额外注入 `routing_hint` 取决于配置

## 11. Offline / Online Strategy Training

相关入口在：

- [strategy_training_runner.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/strategy_training/strategy_training_runner.py)
- [online_strategy_training_runner.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/strategy_training/online_strategy_training_runner.py)

### 11.1 offline

offline training 的基本思路是：

1. 跑 QA
2. 记录 attempt / tool chain / correctness
3. 抽 query pattern
4. 提取 effective tool chain
5. distill 成 strategy template
6. 写入策略库

### 11.2 online

online training 是边答题边积累：

- real traces
- failure reflections
- synthetic QA
- online buffer

当前 agent 初始化时已经支持 online learning 组件异步化，但是否启用取决于 config。

## 12. QUALITY 实验里的真实运行方式

QUALITY 评测的关键逻辑在 [run_quality_benchmark.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/run_quality_benchmark.py)。

### 12.1 每篇文章的 workspace

每篇文章会有独立 workspace，里面至少应有：

- `build_marker.json`
- `knowledge_graph/`
- `narrative_graph/`
- `interactions/interaction_results.json`
- `interactions/interaction_records_list.json`
- `sql/Interaction.db`
- `vector_store/`

### 12.2 评测前的重建与校验

benchmark 会做几件重要的事：

1. `_workspace_missing_artifacts()` 检查关键产物是否齐全
2. `_ensure_workspace_vector_stores_current()` 保证 vector store 与 `doc2chunks/all_document_chunks` 同步
3. 通过 `_build_article_config()` 把 graph/vector/sql 路径指向当前文章 workspace

### 12.3 MCQ answer adapter

QUALITY 的 evaluator 不只是“把 agent 原话拿来判分”，还会做多层后处理：

- choice extractor
- open-answer to choice adapter
- low-confidence fallback
- terminal MCQ enforcement
- posthoc choice recovery

其中在 `no_strategy_agent` 下，还会额外在必要时调用：

- `choice_grounded_evidence_search`

来补做逐选项证据比较。

所以当前 QUALITY 结果，是“agent + answer adapter + choice recovery”的完整系统结果。

## 13. 当前 stable 配置

当前常用 QUALITY 配置见 [config_openai_quality_stable.yaml](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/configs/config_openai_quality_stable.yaml)。

它的关键特征是：

- `aggregation_mode: narrative`
- `graph_store_path: knowledge_graph/graph_runtime_langgraph.pkl`
- `vector_store_type: chroma`
- `strategy_memory.enabled: false`
- `strategy_memory.read_enabled: true`
- `runtime_routing_note_enabled: true`
- `runtime_router_mode: legacy_full`
- `hidden_tool_names: [search_related_content]`

也就是说，这个 stable 配置经常表现为：

- 不做正式 offline strategy 命中
- 但仍可能启用 router / routing note / runtime read 逻辑

具体某次 benchmark 是否关闭这些能力，还要看运行时参数有没有覆盖 config。

## 14. 当前系统和旧版的本质差异

和旧 `NarrativeKnowledgeWeaver` 相比，现在系统最大的变化是：

- 不依赖 Neo4j
- 主图是 JSON + 本地 `graph_runtime_langgraph.pkl`
- graph load 是显式加载，不是外部图数据库持久连接
- 向量库是本地 `Chroma`
- interaction 是 JSON + SQLite
- QA 侧更强调 composite retrieval 和 runtime routing

换句话说，现在系统是一个“本地 workspace 自闭环”的文章级 QA 系统，而不是一个“中心化 Neo4j 服务”。

## 15. 你真正可以怎么理解它

如果只用一句话概括当前系统：

它先把文章编译成一个可复用的本地检索 workspace，再让 retrieval agent 在图、向量、BM25、interaction SQL 和 narrative 聚合层之间选择合适工具去回答问题。

更具体一点：

- 抽取阶段负责把原始文本编译成结构化资产
- builder 负责把这些资产变成图、向量、SQL 和 narrative 层
- retriever agent 负责在这些资产之上调工具
- evaluator 负责把自由文本答案收束成可判分的 MCQ 结果

## 16. 相关代码入口

如果你之后要继续改系统，建议优先看这些文件：

- [graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/graph_builder.py)
- [narrative_graph_builder.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/narrative_graph_builder.py)
- [retriever_agent.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent.py)
- [retriever_agent_langchain.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent_langchain.py)
- [tool_router.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retrieval/tool_router.py)
- [composite_tools.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/functions/tool_calls/composite_tools.py)
- [retrieval_strategy_memory.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/memory/retrieval_strategy_memory.py)
- [strategy_training_runner.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/strategy_training/strategy_training_runner.py)
- [online_strategy_training_runner.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/strategy_training/online_strategy_training_runner.py)
- [run_quality_benchmark.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/run_quality_benchmark.py)

