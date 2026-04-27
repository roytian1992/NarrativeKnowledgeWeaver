# 当前 Agent 运行说明：从知识图谱构建到问答

本文档以当前仓库 `NarrativeWeaver` 的代码为准，说明现在这套 Agent 是如何从原始叙事文本出发，先构建可复用资产，再执行检索式问答的。

重点覆盖四件事：

1. 离线阶段如何把原始 JSON 编译成可复用的 workspace 资产
2. 这些资产在磁盘上长什么样
3. 在线问答时 `QuestionAnsweringAgent` 如何装载工具并运行
4. 现在实验里常见的运行方式和当前推荐配置在哪里


## 1. 总体结构

当前系统是一个两阶段流程：

1. 离线构建
   - 原始叙事文本先被切块、抽取实体关系、补属性、抽 interaction、构建文档超级节点
   - 然后进一步构建 narrative graph，包括 `Episode`、`Storyline` 及其关系
   - 最后把这些 JSON 资产重新加载进本地图运行时图，并补齐向量索引与 centrality
2. 在线问答
   - `QuestionAnsweringAgent` 读取已经构建好的 workspace
   - 组装图工具、文本检索工具、narrative 检索工具、可选 SQL 工具
   - 把问题交给 retrieval runtime
   - runtime 在多轮工具调用中收集证据并输出答案

核心入口：

- 单文件全量构图入口：`main.py`
- 统一 QA 门面：`core/agent/retriever_agent.py`
- 当前默认 QA backend：`core/agent/retriever_agent_langchain.py`
- 当前 runtime 实现：`core/agent/retrieval/langgraph_runtime.py`


## 2. 离线构图阶段

### 2.1 两种常见入口

现在仓库里离线构图主要有两种跑法。

#### A. 直接跑 `main.py`

文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/main.py`

它适合单篇输入、一次性全量构图。主流程是：

1. `KnowledgeGraphBuilder.prepare_chunks()`
2. `extract_entity_and_relation()`
3. `run_extraction_refinement()`
4. `build_entity_and_relation_basic_info()`
5. `postprocess_and_save()`
6. `extract_properties()`
7. `extract_interactions()`
8. `store_interactions_to_sql()`
9. `build_doc_entities()`
10. `load_json_to_graph_store()`
11. `NarrativeGraphBuilder.extract_episodes()`
12. `extract_episode_relations()`
13. `break_episode_cycles()`
14. `build_storyline_candidates()`
15. `extract_storylines_from_candidates()`
16. `extract_storyline_relations()`
17. `NarrativeGraphBuilder.load_json_to_graph_store()`

#### B. 通过 benchmark 脚本按文章构建 workspace

典型文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/run_quality_benchmark.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_benchmark.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`

这条路不是直接把所有东西写到 `data/`，而是给每篇文章建立独立 workspace，然后优先复用已有资产。

在 `QUALITY` 中，核心封装是：

- `ensure_article_ready()`
- `_build_article_workspace()`
- `load_existing_article_workspace_or_raise()`

它们的逻辑是：

1. 先看 workspace 里是否已有 `build_marker.json`
2. 再检查关键资产是否齐全
3. 如果图运行时已经完整，直接复用
4. 如果 JSON 资产还在但运行时图不完整，则重新加载到 graph store
5. 只有在缺资产时才重新离线构建

这就是现在实验可以“复用图资产”的基础。


## 3. 基础知识图谱如何构建

核心文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/builder/graph_builder.py`

### 3.1 切块与文本索引

`prepare_chunks()` 负责把输入 JSON 规范化成检索与抽取都能使用的中间格式。

它会生成稳定 ID：

- `raw_doc_id`
- `doc_segment_id`
- `document_id`
- `chunk_id`

典型输出：

- `all_document_chunks.json`
- `doc2chunks.json`
- `doc2chunks_index.json`

如果配置允许，这一步还会顺手准备文档级/句子级向量库所需的文本材料。

### 3.2 实体关系抽取

`extract_entity_and_relation()` 会对切块后的文本执行实体和关系抽取。随后：

- `run_extraction_refinement()` 做抽取结果清洗与归并
- `build_entity_and_relation_basic_info()` 生成基础实体/关系信息

这一阶段产出的核心 JSON 会成为后续 refined graph 的输入。

### 3.3 图后处理与基础图生成

`postprocess_and_save()` 会把 refined 结果组装成真正的基础图。

关键输出：

- `relation_info_refined.json`
- `graph_nx.pkl`

这里用的是本地 `networkx.MultiDiGraph`，不是 Neo4j。

### 3.4 重要节点属性抽取

`extract_properties()` 会基于图结构、中心性和上下文边描述，为重要节点补属性。

关键输出：

- `entity_info_refined.json`

这一步保留了我们一直强调的“基于重要节点做属性抽取”的核心设计。

### 3.5 interaction 抽取与 SQL 化

相关函数：

- `extract_interactions()`
- `store_interactions_to_sql()`

典型输出：

- `interactions/interaction_results.json`
- `interactions/interaction_records_list.json`
- `sql/Interaction.db`

这里 SQL 主要是 interaction 的存储层，不是主图数据库。

### 3.6 文档超级节点

`build_doc_entities()` 会构造 section/document 超级节点，并把它们和普通实体用 `CONTAINS` 类关系连起来。

关键输出：

- `doc_entities.json`
- `doc_entity_edges.json`

这样 QA 时就能把“文档结构节点”和“普通实体节点”放在同一个图运行时里检索。

### 3.7 加载到本地图运行时图

`load_json_to_graph_store()` 会把以下资产统一写回当前 graph runtime：

- refined entities
- refined relations
- doc entities
- doc-entity edges

写入之后还会继续做几件事：

1. 建立图上的向量索引
2. 为实体补 embedding
3. 统一 superlabel
4. 计算 centrality

也就是说，问答真正依赖的不是零散 JSON，而是“JSON + runtime graph + embedding cache”这一整套状态。


## 4. Narrative Graph 如何构建

核心文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/builder/narrative_graph_builder.py`

### 4.1 P1：Episode 抽取

`extract_episodes()` 会从文档节点出发，把局部叙事内容提升成 `Episode`。

这一步还会生成 episode support edges，并可选补 episode embedding。

常见输出位于 `narrative_graph/` 下，例如：

- `global/episodes.json`
- `global/episode_support_edges.json`
- `global/episode_packs.json`

### 4.2 P2：Episode 之间关系

`extract_episode_relations()` 会为 episode 配对找候选，再抽取 episode-relation。

当前候选判定不是简单全连接，而是依赖：

- primary anchors
- context anchors
- embedding similarity
- top-k 截断

相关参数来自：

- `config.narrative_graph_builder`

典型输出：

- `global/episode_relations.json`
- `episodes/candidate_pairs.json`

### 4.3 去环与 storyline 构建

后续阶段依次是：

1. `break_episode_cycles()`
2. `build_storyline_candidates()`
3. `extract_storylines_from_candidates()`
4. `extract_storyline_relations()`

典型输出：

- `global/storylines.json`
- `global/storyline_support_edges.json`
- `global/storyline_relations.json`
- `global/episode_relations_dag.json`

### 4.4 Narrative 资产写回运行时图

`NarrativeGraphBuilder.load_json_to_graph_store()` 会把：

- `Episode`
- episode support edges
- episode relations
- `Storyline`
- storyline support edges
- storyline relations

加载进同一个本地图运行时图。

因此，问答阶段看到的是“基础图 + narrative 聚合图”的合体。


## 5. 典型 workspace 长什么样

当前实验通常按文章维护独立 workspace。一个完整 workspace 大致会有这些目录：

- `knowledge_graph/`
- `narrative_graph/`
- `vector_store/`
- `sql/`
- `interactions/`
- `build_marker.json`

`build_marker.json` 是资产复用的锚点。它记录这篇文章已经构建过哪些离线资产，并帮助 benchmark 决定是：

- 直接复用
- 只重载 runtime
- 还是必须重建


## 6. QA Agent 如何初始化

统一入口：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retriever_agent.py`

这个文件本身只是 facade。它会根据 backend 选择真正实现。

当前默认 backend 是：

- `langchain`

真正主要实现文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retriever_agent_langchain.py`

`QuestionAnsweringAgent.__init__()` 会初始化这些关键对象：

1. `GraphStore`
2. `GraphQueryUtils`
3. `OpenAILLM` 多套 profile
4. `DocumentParser`
5. 文档级 / 句子级向量检索组件
6. `RetrievalToolRouter`
7. 全部工具集合
8. retrieval runtime

因此现在的 agent 不是“prompt + tools 的薄壳”，而是一个完整运行时对象图。


## 7. QA 时有哪些工具

工具来源主要有三类：

1. 图工具
   - 例如实体检索、关系查询、centrality、section 关联等
2. 文本检索工具
   - BM25
   - 向量检索
   - section 证据检索
   - narrative hierarchical 检索
3. 可选 SQL 工具
   - interaction 相关

工具最终通过 `QuestionAnsweringAgent._all_tools()` 汇总，再按当前实验 profile 做隐藏/放行。

注意：不同 benchmark 往往会再叠加自己的工具可见性策略，而不是所有数据集都共用一套可见工具。


## 8. 当前 runtime 怎么运行

核心文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`

### 8.1 两个 runtime 版本

这个文件里有两个 runtime：

1. `LangGraphAssistantRuntime`
   - 基础版 JSON tool loop
2. `S4LangGraphAssistantRuntime`
   - 结构化 planner/evaluator/finalizer 版

工厂函数：

- `create_langgraph_assistant_runtime()`

默认代码级 fallback 是 `s3` 风格基础 runtime；但最近实验常用的是显式传入：

- `assistant_runtime_variant: s4`

所以如果文档里说“当前实验主线 runtime”，通常指的是 `S4LangGraphAssistantRuntime`。

### 8.2 基础版 runtime 的执行方式

基础版 runtime 采用手写 JSON-only tool calling。

模型只能返回两种核心结构：

```json
{"tool_name":"<tool>","tool_arguments":{"arg":"value"}}
```

或

```json
{"tool_calls":[{"tool_name":"<tool>","tool_arguments":{"arg":"value"}}]}
```

最终回答则是：

```json
{"final_answer":"..."}
```

执行图是：

1. `model`
2. 如果有工具调用则进入 `tools`
3. 工具结果转成 `ToolMessage`
4. 回到 `model`
5. 直到没有新工具调用或达到预算

### 8.3 工具预算

runtime 把预算分成两层：

1. 总轮数
   - `max_tool_rounds_per_run`
2. 每轮最多多少工具
   - 首轮：`first_round_max_tool_calls`
   - 后续轮：`followup_round_max_tool_calls`

也就是说，“最多三轮，每轮最多几个工具”是 runtime 内部硬约束，不是提示词里说说而已。

### 8.4 并发工具执行

同一轮如果模型发起多个工具，runtime 会用线程池并发执行：

- `parallel_tool_workers`

这让首轮“多证据并发探测”成为可能。


## 9. S4 runtime 比基础版多了什么

`S4LangGraphAssistantRuntime` 在基础 JSON tool loop 之上，又加入了几层结构化状态：

- `evidence_pool`
- `curated_evidence_pool`
- `entities_found`
- `plan_notes`
- `evaluation_notes`
- `draft_answer`
- `stagnation_count`

它的关键改动是：

1. 不只保存原始工具返回，还维护结构化 evidence pool
2. 支持 evidence curation
3. 能区分 first round 和 follow-up round 的工具空间
4. 会根据当前证据缺口决定下一轮还要不要继续查

这就是为什么现在 QA 不再是“模型看完一堆 raw tool output 直接回答”，而是“多轮收集证据后再决策”。


## 10. 当前实验里常见的 QA 策略

### 10.1 FiarytableQA 当前默认 profile

当前最明确、最成体系的一套可见工具策略，是 FiarytableQA 这边的 profile 文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/task_specs/tool_visibility/fiarytableqa_agent_default.json`

它目前定义的关键运行参数是：

- `assistant_runtime_variant = s4`
- `bypass_runtime_tool_router = true`
- `max_tool_rounds_per_run = 3`
- `first_round_max_tool_calls = 2`
- `followup_round_max_tool_calls = 3`
- `parallel_tool_workers = 4`

含义是：

1. 让 assistant 直接在“当前可见工具集”上自规划
2. 不再先走外部 runtime router 做分支裁剪
3. 总共最多 3 个工具轮次
4. 首轮最多 2 个模型自选工具
5. 但首轮可见工具会被额外限制在 profile 指定的小集合内

这个 profile 里当前首轮可见工具是：

- `bm25_search_docs`
- `vdb_search_sentences`
- `section_evidence_search`

而 plannable tools 则比首轮更大，后续轮可以继续访问 narrative / entity / interaction 等工具。

### 10.2 QUALITY 和 STAGE

`QUALITY` 与 `STAGE` 也会在 benchmark 脚本里额外叠加隐藏工具集合。

具体位置：

- `QUALITY`：`run_quality_benchmark.py`
- `STAGE`：`run_stage_task2_benchmark.py`

这两个脚本通常负责：

1. 定义 benchmark 专属 hidden tools
2. 控制是否启用 SQL
3. 决定是走默认 agent，还是只允许 hybrid rag 工具子集


## 11. 一次问答是怎么跑完的

从调用 `agent.ask(question)` 开始，一次单题流程可以概括为：

1. `QuestionAnsweringAgent.ask()`
2. 准备 `memory_ctx`
3. 生成 runtime system message
4. 根据当前 profile 和 hidden tools 决定 assistant 能看到哪些工具
5. 创建或复用 assistant runtime
6. runtime 首轮调用若干工具取证
7. 工具结果进入 `ToolMessage` 和 evidence pool
8. 如果证据不足，再进入 follow-up round
9. 直到：
   - 已有足够证据
   - 或用完 tool-round budget
10. 输出最终答案 JSON / 文本
11. benchmark 外层再做答案抽取、适配和 judge

如果是评测脚本，还会继续做：

1. 记录 latency
2. 保存 tool trace
3. 调 judge LLM 判分
4. 汇总多 pass 结果


## 12. 资产复用是怎么实现的

当前“复用已经构好的图资产”主要依赖 benchmark 层，而不是 `main.py` 自己。

核心机制：

1. 每篇文章有独立 workspace
2. workspace 下有 `build_marker.json`
3. benchmark 先检查关键资产是否齐全
4. 如果 runtime graph 还在且完整，直接复用
5. 如果 runtime graph 缺失但 JSON 资产仍在，则只做 reload
6. 只有缺核心资产时才重建

所以现在最推荐的实验习惯是：

- 构图阶段和问答阶段都围绕 per-article workspace 做
- 不要把“只在内存里存在的图”当成长期资产


## 13. 典型命令

### 13.1 单文档全量构图

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
python main.py \
  --json_file /path/to/article.json \
  --config configs/config_openai.yaml
```

### 13.2 QUALITY 按文章构图并评测

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
python experiments/quality/run_quality_benchmark.py ...
```

`QUALITY` 会自动：

- 建立 per-article workspace
- 构图
- 复用已有资产
- 初始化 agent
- 执行多次问答

### 13.3 STAGE Task 2

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
python experiments/stage/run_stage_task2_benchmark.py ...
```

### 13.4 FiarytableQA

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
python experiments/fiarytableqa/run_fiarytable_benchmark.py ...
```

Fiary 的脚本会把当前 profile 的 `rag_cfg` 和 `hidden_tool_names` 一起灌给 `QuestionAnsweringAgent`。


## 14. 现在这套 Agent 的关键特征

如果只抓最关键的点，现在这套 Agent 可以概括成：

1. 底座仍然是“先离线构图，再在线检索问答”
2. 主图已经是本地 `networkx + graph runtime`，不是 Neo4j
3. narrative aggregation 仍然是主链的一部分
4. 重要节点属性抽取没有被砍掉
5. interaction 作为独立 SQL 侧资产保留
6. QA 不再是单次生成，而是多轮 JSON tool loop
7. 当前实验主线更接近 `S4 runtime + 小工具集首轮探测 + follow-up 补证据`
8. benchmark 已经支持按文章复用 workspace 与构图资产


## 15. 最值得先看的代码

如果要快速接手，建议先按这个顺序读：

1. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/main.py`
2. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/builder/graph_builder.py`
3. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/builder/narrative_graph_builder.py`
4. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retriever_agent.py`
5. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retriever_agent_langchain.py`
6. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
7. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/run_quality_benchmark.py`
8. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_benchmark.py`
9. `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`


## 16. 一句话总结

现在这套 Agent 的真实运行方式，不是“直接把文章喂给 LLM”，而是：

先把文章编译成基础图、叙事图、向量索引和 interaction 资产，再让一个受工具预算约束的 retrieval agent 在这些资产上多轮取证，最后输出答案。
