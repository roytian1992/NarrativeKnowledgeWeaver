# NarrativeKnowledgeWeaver

NarrativeKnowledgeWeaver is a narrative analysis and retrieval framework built around a JSON-first knowledge extraction pipeline, a local NetworkX runtime graph, Chroma vector stores, SQLite interaction storage, and a tool-calling retriever agent.

The current codebase supports:
- base knowledge graph extraction from screenplay / novel / general documents
- post-refinement property extraction
- interaction extraction to JSON plus optional SQLite import
- two aggregation families on top of the base graph:
  - `narrative`: `Event -> Episode -> Storyline`
  - `community`: Leiden communities plus GraphRAG-style community reports
  - `full`: build both
- a retrieval-first QA agent that loads tools according to aggregation mode
- two separate memory systems:
  - extraction memory for the KG pipeline
  - strategy memory for retrieval-time tool routing

This README reflects the current repository state. Older descriptions about the architecture may no longer match the code.

## What The Repository Does

### 1. Base knowledge graph pipeline
The base pipeline is implemented by [`core/builder/graph_builder.py`](core/builder/graph_builder.py).

It performs the following steps:
1. split the input narrative into chunks
2. extract entities and relations into JSON
3. refine and merge graph results
4. build entity / relation basic info files
5. extract properties after refinement
6. extract interactions after property extraction
7. dump interaction JSON files to `data/narrative_interactions`
8. optionally import interaction records into SQLite
9. build document-level nodes and load the graph into the local runtime graph store

### 2. Narrative aggregation
The narrative aggregation pipeline is implemented by [`core/builder/narrative_graph_builder.py`](core/builder/narrative_graph_builder.py).

It builds:
- Episode nodes and `EPISODE_CONTAINS` support edges
- Episode causal / semantic relations
- Storyline candidates and Storyline nodes
- Storyline support edges and relations

Intermediate and merged outputs are stored under `data/narrative_graph/`.

### 3. Community aggregation
The community aggregation pipeline is implemented by [`core/builder/community_graph_builder.py`](core/builder/community_graph_builder.py).

It:
- runs local community detection on the existing runtime graph
- writes community assignments back to the runtime graph
- creates community reports with LLM summaries
- stores community report embeddings in `data/vector_store/community`

Outputs are stored under `data/community_graph/`.

### 4. Retrieval agent
The retrieval agent is implemented by [`core/agent/retriever_agent.py`](core/agent/retriever_agent.py).

It is a single-turn, retrieval-first agent that can combine:
- graph tools from the local runtime graph
- vector retrieval tools
- BM25 / document mapping utilities
- optional SQLite interaction tools
- composite retrieval tools such as narrative hierarchical search and community GraphRAG search

The loaded tool set depends on `global.aggregation_mode`:
- `narrative`: load Episode / Storyline tools
- `community`: load Community tools
- `full`: load both

### 5. Memory
There are two different memory subsystems:

- `extraction_memory`
  - used during KG extraction
  - stores raw / distilled / realtime extraction memories
  - located under `data/memory/raw_memory`, `data/memory/distilled_memory`, `data/memory/realtime_memory`

- `strategy_memory`
  - used by the retriever agent at read time
  - trained offline from question-answer CSV files
  - runtime library stored at `data/memory/strategy/strategy_library.json`
  - training runner implemented in [`core/strategy_training/strategy_training_runner.py`](core/strategy_training/strategy_training_runner.py)

## Repository Layout

```text
NarrativeKnowledgeWeaver/
├── configs/                     # YAML configs
├── core/
│   ├── agent/                   # extraction / retrieval agents
│   ├── builder/                 # graph builders and managers
│   ├── functions/               # tool calls, memory functions, aggregation functions
│   ├── memory/                  # extraction memory and strategy memory
│   ├── model_providers/         # LLM / embedding / rerank wrappers
│   ├── storage/                 # local graph store, vector store, SQL store
│   └── utils/                   # config, formatting, graph helpers, general utils
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
├── reports/                     # tool tests, smoke reports, QA reports
├── strategy_training/           # offline strategy-memory training runs
├── task_specs/                  # prompts, schemas, task settings, tool metadata
├── main.py                      # base graph + configured aggregation pipeline
└── test_main.py                 # community-only test entry
```

## Prompt / Schema Management

Prompt, schema, and task-setting files are managed under `task_specs/`.

Language-dependent resources are selected by [`core/utils/config.py`](core/utils/config.py):
- `global.language: zh`
  - `task_specs/prompts`
  - `task_specs/task_settings`
  - `task_specs/schema`
  - `task_specs/tool_metadata/zh`
- `global.language: en`
  - `task_specs/prompts_en`
  - `task_specs/task_settings_en`
  - `task_specs/schema_en`
  - `task_specs/tool_metadata/en`

This is the current source of truth. The old `core/prompts` style is no longer the active organization.

## Requirements

- Python 3.12
- a compatible OpenAI-style LLM endpoint for generation
- embedding endpoint
- reranker endpoint if reranking is enabled

## Installation

### Conda

The provided environment file uses the environment name `screenplay`.

```bash
conda env create -f environment.yml
conda activate screenplay
```

### Pip

```bash
pip install -r requirements.txt
```

## Configuration Overview

The main config file is parsed by [`core/utils/config.py`](core/utils/config.py).

Important sections:
- `global`
  - `language`: `zh` or `en`
  - `doc_type`: `screenplay`, `novel`, or `general`
  - `aggregation_mode`: `narrative`, `community`, or `full`
- `document_processing`
  - chunk sizes for graph extraction, vector storage, sentence storage, and BM25 subchunking
- `knowledge_graph_builder`
  - output path and extraction concurrency
- `storage`
  - local graph store path, vector store path, SQL path
- `llm`, `embedding`, `rerank`
  - model endpoints
- `narrative_graph_builder`
  - episode / storyline construction settings
- `community_graph_builder`
  - Leiden / community report settings
- `extraction_memory`
  - memory used by extraction agents
- `strategy_memory`
  - runtime strategy-template retrieval

Example config files:
- [`configs/config_openai.yaml`](configs/config_openai.yaml)
- [`configs/config_local.yaml`](configs/config_local.yaml)

## Running The Pipeline

### End-to-end build

`main.py` always runs the base graph pipeline first, then executes the aggregation pipeline selected by `global.aggregation_mode`.

```bash
python main.py \
  --config configs/config_openai.yaml \
  --json_file examples/documents/wandering_earth2.json
```

Current `main.py` behavior:
- base graph pipeline is always executed
- then:
  - `narrative` -> build Episode / Storyline aggregation
  - `community` -> build community aggregation
  - `full` -> build narrative first, then community

### Community-only rebuild on an existing graph

If you already have the base runtime graph and only want to rebuild communities:

```bash
python test_main.py \
  --config configs/config_openai.yaml \
  --clear_previous_community
```

Optional flags:
- `--clear_previous_community`
- `--clear_previous_narrative`

## Interaction Extraction And SQL Storage

Interaction extraction is not stored in the knowledge graph.

Current behavior:
- extraction happens after property extraction
- source input is the refined extraction result plus Character / Object candidates
- outputs are dumped to `data/narrative_interactions`
- SQL import is a separate step

Relevant code:
- extraction: [`KnowledgeGraphBuilder.extract_interactions()`](core/builder/graph_builder.py)
- SQL import: [`KnowledgeGraphBuilder.store_interactions_to_sql()`](core/builder/graph_builder.py)
- generic SQLite persistence: [`core/storage/sql_store.py`](core/storage/sql_store.py)

Current interaction files:
- `data/narrative_interactions/interaction_results.json`
- `data/narrative_interactions/interaction_records_list.json`

Current SQL target:
- default DB: `data/sql/Interaction.db`
- default table: `Interaction_info`

## Tool System

Tool implementations live in [`core/functions/tool_calls/`](core/functions/tool_calls).

Current categories:
- `graphdb_tools.py`
  - entity lookup, section search, neighborhood search, episode / storyline / community search
- `vectordb_tools.py`
  - document, sentence, hierarchical, and document-id based retrieval
- `native_tools.py`
  - BM25, document-id/title mapping, section content filtering
- `sqldb_tools.py`
  - dialogue and interaction lookup from SQLite
- `composite_tools.py`
  - `narrative_hierarchical_search`
  - community GraphRAG retrieval
  - section-level evidence search

The retriever agent loads a subset of these tools depending on aggregation mode and whether SQL tools are enabled.

## Using The Retriever Agent Programmatically

There is no single dedicated CLI for QA in the current repository. The simplest way to use the agent is from Python.

```python
from core import KAGConfig
from core.agent.retriever_agent import QuestionAnsweringAgent

config = KAGConfig.from_yaml("configs/config_openai.yaml")
agent = QuestionAnsweringAgent(
    config,
    aggregation_mode="narrative",   # or community / full
    enable_sql_tools=True,
)

responses = agent.ask("550系列有哪几个型号？分别出现在了哪些场次中？", lang="zh")
print(agent.extract_final_text(responses))
agent.close()
```

## Strategy Memory Training

Strategy memory is trained offline from question-answer CSV files.

Current implementation:
- trainer: [`StrategyMemoryTrainingRunner`](core/strategy_training/strategy_training_runner.py)
- runtime reader: [`RetrievalStrategyMemory`](core/memory/retrieval_strategy_memory.py)

The trainer currently supports:
- multiple attempts per question
- answer judging
- effective tool-chain extraction
- failure reflection
- retry instruction generation
- retry attempts
- template clustering and deduplication
- runtime library export

Example:

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

# optional: clear runtime strategy library first
runner.clear_runtime_library()
result = runner.run(reset_runtime_library=True)
runner.close()
print(result)
```

Training artifacts are stored under `strategy_training/<dataset_name>/`.
The runtime strategy library is stored at `data/memory/strategy/strategy_library.json`.

## Main Output Directories

- `data/knowledge_graph/`
  - chunked documents, extraction results, refined entities / relations, basic info files
- `data/narrative_interactions/`
  - interaction JSON outputs
- `data/sql/`
  - SQLite databases
- `data/narrative_graph/`
  - episodes, storylines, support edges, relations
- `data/community_graph/`
  - community assignments, hierarchy, reports
- `data/vector_store/document/`
  - document-level vector store
- `data/vector_store/sentence/`
  - sentence-level vector store
- `data/vector_store/community/`
  - community summary vector store
- `data/memory/`
  - extraction memory and runtime strategy library
- `reports/`
  - tool tests, smoke tests, QA reports

## Notes And Limitations

- `main.py` is the supported entry point for end-to-end builds.
- `test_main.py` is a utility entry for community-only rebuilds on top of an existing graph.
- The retriever agent is currently single-turn.
- Strategy-memory training is currently serial at the question/attempt loop level; large runs can take a long time.
- Some training-time substeps, such as effective tool-chain extraction, can still hit long-context limits if a single attempt becomes very large.
- SQL tools are optional and are not loaded unless `enable_sql_tools=True`.

## Related Files

- [`README_zh.md`](README_zh.md)
- [`main.py`](main.py)
- [`test_main.py`](test_main.py)
- [`core/agent/retriever_agent.py`](core/agent/retriever_agent.py)
- [`core/builder/graph_builder.py`](core/builder/graph_builder.py)
- [`core/builder/narrative_graph_builder.py`](core/builder/narrative_graph_builder.py)
- [`core/builder/community_graph_builder.py`](core/builder/community_graph_builder.py)
