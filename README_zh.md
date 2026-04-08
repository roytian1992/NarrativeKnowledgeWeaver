# NarrativeKnowledgeWeaver

NarrativeKnowledgeWeaver 是一个面向叙事文本的知识抽取、聚合建模与检索问答框架。当前代码库以 JSON-first 的知识图谱构建流程为核心，结合本地 NetworkX 运行时图、Chroma 向量库、SQLite 交互数据库，以及一个基于工具调用的检索型 Agent。

当前仓库已经支持：
- 从 screenplay / novel / general 文档构建基础知识图谱
- 基于 refined graph 的属性抽取
- 交互信息抽取到 JSON，并可选择导入 SQLite
- 两类聚合建模：
  - `narrative`：`Event -> Episode -> Storyline`
  - `community`：Leiden 社区划分 + GraphRAG 风格社区摘要
  - `full`：两者都构建
- 一个按 aggregation mode 加载工具的检索型 QA Agent
- 两套彼此独立的 memory：
  - 知识抽取记忆
  - 检索策略记忆

这份 README 以当前仓库真实实现为准，旧版文档中的部分架构描述已经过时。

## 当前功能概览

### 1. 基础知识图谱构建
基础流水线由 [`core/builder/graph_builder.py`](core/builder/graph_builder.py) 实现。

当前流程如下：
1. 文本切块
2. 实体与关系抽取
3. 图谱结果 refinement / merge
4. 生成 entity / relation basic info
5. 在 refinement 之后进行属性抽取
6. 在属性抽取之后进行 interaction 抽取
7. interaction 结果落盘到 `data/narrative_interactions`
8. 可选地导入 SQLite
9. 构建文档节点并加载到本地运行时图

### 2. Narrative 聚合
Narrative 聚合由 [`core/builder/narrative_graph_builder.py`](core/builder/narrative_graph_builder.py) 实现。

它会构建：
- Episode 节点与 `EPISODE_CONTAINS` 支持边
- Episode 之间的因果 / 语义关系
- Storyline candidate 与 Storyline 节点
- Storyline 支持边与 Storyline 关系

输出保存在 `data/narrative_graph/`。

### 3. Community 聚合
Community 聚合由 [`core/builder/community_graph_builder.py`](core/builder/community_graph_builder.py) 实现。

它会：
- 在现有运行时图上做本地社区划分
- 将社区归属写回运行时图
- 用 LLM 生成社区报告 / 摘要
- 将社区摘要向量存入 `data/vector_store/community`

输出保存在 `data/community_graph/`。

### 4. 检索型 Agent
检索 Agent 由 [`core/agent/retriever_agent.py`](core/agent/retriever_agent.py) 实现。

它是一个单轮、retrieval-first 的 Agent，可以组合：
- 本地运行时图工具
- 向量检索工具
- BM25 与 document mapping 工具
- 可选的 SQLite interaction 工具
- 组合检索工具，例如 narrative hierarchical search 与 community GraphRAG search

它加载哪些工具，取决于 `global.aggregation_mode`：
- `narrative`：加载 Episode / Storyline 相关工具
- `community`：加载 Community 相关工具
- `full`：两类都加载

### 5. 两类 Memory
当前仓库里有两套独立的 memory：

- `extraction_memory`
  - 用于知识图谱抽取阶段
  - 保存 raw / distilled / realtime memory
  - 位于 `data/memory/raw_memory`、`data/memory/distilled_memory`、`data/memory/realtime_memory`

- `strategy_memory`
  - 用于检索 Agent 运行时的策略提示
  - 离线训练，在线读取
  - 运行时策略库位于 `data/memory/strategy/strategy_library.json`
  - 训练入口见 [`core/strategy_training/strategy_training_runner.py`](core/strategy_training/strategy_training_runner.py)

## 仓库结构

```text
NarrativeKnowledgeWeaver/
├── configs/                     # YAML 配置
├── core/
│   ├── agent/                   # 抽取 / 检索 agent
│   ├── builder/                 # graph builder 与 manager
│   ├── functions/               # tool calls、memory functions、aggregation functions
│   ├── memory/                  # extraction memory 与 strategy memory
│   ├── model_providers/         # LLM / embedding / rerank 封装
│   ├── storage/                 # 本地图存储、vector store、SQL store
│   └── utils/                   # config、format、图工具、general utils
├── data/
│   ├── knowledge_graph/
│   ├── narrative_graph/
│   ├── community_graph/
│   ├── narrative_interactions/
│   ├── vector_store/
│   ├── sql/
│   └── memory/
├── examples/
│   ├── documents/
│   └── datasets/
├── reports/                     # tool 测试、烟测、QA 报告
├── strategy_training/           # 策略记忆训练结果
├── task_specs/                  # prompts、schemas、task settings、tool metadata
├── main.py                      # 基础图谱 + aggregation 主入口
└── test_main.py                 # community-only 测试入口
```

## Prompt / Schema 管理

当前 prompts、schemas、task settings 都统一放在 `task_specs/` 下。

语言由 [`core/utils/config.py`](core/utils/config.py) 中的 `global.language` 控制：
- `zh`
  - `task_specs/prompts`
  - `task_specs/task_settings`
  - `task_specs/schema`
  - `task_specs/tool_metadata/zh`
- `en`
  - `task_specs/prompts_en`
  - `task_specs/task_settings_en`
  - `task_specs/schema_en`
  - `task_specs/tool_metadata/en`

这才是当前代码实际使用的路径。旧版文档里 `core/prompts` 那种说法已经不再准确。

## 环境要求

- Python 3.12
- 一个兼容 OpenAI 风格 API 的生成模型服务
- embedding 服务
- 如果启用 rerank，需要 reranker 服务

## 安装

### Conda

仓库里的 `environment.yml` 当前环境名为 `screenplay`。

```bash
conda env create -f environment.yml
conda activate screenplay
```

### Pip

```bash
pip install -r requirements.txt
```

## 配置说明

配置由 [`core/utils/config.py`](core/utils/config.py) 解析。

当前比较关键的配置段：
- `global`
  - `language`: `zh` 或 `en`
  - `doc_type`: `screenplay`、`novel`、`general`
  - `aggregation_mode`: `narrative`、`community`、`full`
- `document_processing`
  - 控制图谱切块、句子级切块、BM25 子切块
- `knowledge_graph_builder`
  - 基础图谱输出路径与抽取并发
- `storage`
  - 本地图存储、向量库、SQL 路径
- `llm`、`embedding`、`rerank`
  - 模型服务配置
- `narrative_graph_builder`
  - Episode / Storyline 构建参数
- `community_graph_builder`
  - Leiden / community report 参数
- `extraction_memory`
  - 知识抽取记忆
- `strategy_memory`
  - 运行时策略记忆

示例配置：
- [`configs/config_openai.yaml`](configs/config_openai.yaml)
- [`configs/config_local.yaml`](configs/config_local.yaml)

## 如何运行

### 端到端构建

`main.py` 当前的行为是：先跑基础知识图谱流水线，再根据 `global.aggregation_mode` 跑对应的 aggregation。

```bash
python main.py \
  --config configs/config_openai.yaml \
  --json_file examples/documents/wandering_earth2.json
```

当前 `main.py` 的逻辑：
- 总是先执行 base graph pipeline
- 然后：
  - `narrative` -> 构建 Episode / Storyline
  - `community` -> 构建 Community aggregation
  - `full` -> 先 narrative，再 community

### 基于现有图只重建 community

如果基础图已经在本地运行时图中，只想重建 community：

```bash
python test_main.py \
  --config configs/config_openai.yaml \
  --clear_previous_community
```

可选参数：
- `--clear_previous_community`
- `--clear_previous_narrative`

## Interaction 抽取与 SQL 存储

Interaction 不保存在知识图谱中。

当前实现方式：
- 在 property extraction 之后抽取
- 输入基于 refined extraction 结果 + Character / Object 候选实体
- 先落盘到 `data/narrative_interactions`
- SQL 导入是独立步骤

相关代码：
- 抽取：[`KnowledgeGraphBuilder.extract_interactions()`](core/builder/graph_builder.py)
- 导入 SQL：[`KnowledgeGraphBuilder.store_interactions_to_sql()`](core/builder/graph_builder.py)
- 通用 SQLite 封装：[`core/storage/sql_store.py`](core/storage/sql_store.py)

当前 interaction 输出文件：
- `data/narrative_interactions/interaction_results.json`
- `data/narrative_interactions/interaction_records_list.json`

当前 SQL 默认目标：
- 数据库：`data/sql/Interaction.db`
- 表：`Interaction_info`

## Tool 体系

工具实现位于 [`core/functions/tool_calls/`](core/functions/tool_calls)。

当前分为：
- `graphdb_tools.py`
  - 实体查询、Section 查询、邻域查询、Episode / Storyline / Community 查询
- `vectordb_tools.py`
  - document、sentence、hierarchical、document_id 定位检索
- `native_tools.py`
  - BM25、document_id/title 映射、section 内容过滤
- `sqldb_tools.py`
  - 对话与交互信息的 SQLite 查询
- `composite_tools.py`
  - `narrative_hierarchical_search`
  - community GraphRAG 检索
  - section 级证据检索

Retriever Agent 会根据 aggregation mode 和是否启用 SQL，加载对应子集。

## 以编程方式使用 Retriever Agent

当前仓库没有单独封装一个统一 QA CLI。最直接的方式是通过 Python 调用。

```python
from core import KAGConfig
from core.agent.retriever_agent import QuestionAnsweringAgent

config = KAGConfig.from_yaml("configs/config_openai.yaml")
agent = QuestionAnsweringAgent(
    config,
    aggregation_mode="narrative",   # 或 community / full
    enable_sql_tools=True,
)

responses = agent.ask("550系列有哪几个型号？分别出现在了哪些场次中？", lang="zh")
print(agent.extract_final_text(responses))
agent.close()
```

## 策略记忆训练

策略记忆是离线训练、在线读取的。

当前实现：
- 训练器：[`StrategyMemoryTrainingRunner`](core/strategy_training/strategy_training_runner.py)
- 运行时读取：[`RetrievalStrategyMemory`](core/memory/retrieval_strategy_memory.py)

训练器当前支持：
- 每题多次 attempt
- LLM answer judge
- effective tool chain 抽取
- 失败反思
- retry instruction 生成
- retry attempts
- 模板聚类与合并去重
- 运行时策略库导出

示例：

```python
from core import KAGConfig
from core.strategy_training.strategy_training_runner import StrategyMemoryTrainingRunner

config = KAGConfig.from_yaml("configs/config_openai.yaml")
runner = StrategyMemoryTrainingRunner(
    config=config,
    csv_path="examples/datasets/we2_qa.csv",
    dataset_name="we2_training",
    attempts_per_question=5,
    output_root="strategy_training",
)

# 可选：先清空运行时策略库
runner.clear_runtime_library()
result = runner.run(reset_runtime_library=True)
runner.close()
print(result)
```

训练结果保存在 `strategy_training/<dataset_name>/`。
运行时策略库位于 `data/memory/strategy/strategy_library.json`。

## 主要输出目录

- `data/knowledge_graph/`
  - 文本块、抽取结果、refined entities / relations、basic info
- `data/narrative_interactions/`
  - interaction JSON 输出
- `data/sql/`
  - SQLite 数据库
- `data/narrative_graph/`
  - Episode、Storyline、support edges、relations
- `data/community_graph/`
  - community assignments、hierarchy、reports
- `data/vector_store/document/`
  - document 级向量库
- `data/vector_store/sentence/`
  - sentence 级向量库
- `data/vector_store/community/`
  - community summary 向量库
- `data/memory/`
  - extraction memory 与 runtime strategy library
- `reports/`
  - tool 测试、烟测、QA 报告

## 当前限制与说明

- `main.py` 是当前支持的端到端入口。
- `test_main.py` 是基于现有图做 community-only 重建的辅助入口。
- Retriever Agent 当前是单轮的。
- 策略记忆训练当前在 question / attempt 层面基本是串行的，大规模训练会比较慢。
- 训练时某些子步骤，例如 effective tool chain 抽取，在极长上下文下仍可能遇到上下文长度限制。
- SQL tools 默认不加载，只有 `enable_sql_tools=True` 时才启用。

## 相关文件

- [`README.md`](README.md)
- [`main.py`](main.py)
- [`test_main.py`](test_main.py)
- [`core/agent/retriever_agent.py`](core/agent/retriever_agent.py)
- [`core/builder/graph_builder.py`](core/builder/graph_builder.py)
- [`core/builder/narrative_graph_builder.py`](core/builder/narrative_graph_builder.py)
- [`core/builder/community_graph_builder.py`](core/builder/community_graph_builder.py)
