# QUALITY Experiment Plan

Last updated: 2026-03-18 21:12 Asia/Shanghai

## Scope

- Fixed train split: 10 articles
- Fixed test split: 50 eval articles
- Frozen manifest:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/artifacts/split_manifest_train10_eval50_seed20260318.json`
- Sampling seed: `20260318`

## Experiment 1

- Goal: compare traditional RAG vs our agent with `1-pass` and `5-pass`
- Planned run shape:
  - same frozen 50-article test split
  - `repeats=5`
  - reuse the same offline-trained runtime library from the 10 train articles
- Note:
  - the benchmark run will still emit all existing settings
  - final headline comparison can be selected from those outputs without rerunning

## Experiment 2

- Goal: compare agent modes with `1-pass`
- Compared settings:
  - `offline_strategy_agent`
  - `offline_strategy_subagent`
  - `online_strategy_agent`
  - `online_strategy_subagent`
  - keep `no_strategy_agent` and `traditional_hybrid_rag_bm25` in outputs as anchors
- Planned run shape:
  - same frozen 50-article test split
  - `repeats=1`

## Current Execution Plan

1. Run Experiment 2 first with `repeats=1` on the frozen manifest.
2. Keep all reports, progress JSONs, and runtime artifacts under a dedicated run root.
3. Reuse the offline runtime snapshot for Experiment 1 `repeats=5`.

## Stability Note

- Run `quality_exp2_1pass_train10_eval50_seed20260318_20260318_2016` should be treated as failed / polluted.
- Failure mode was upstream `Connection error` during extraction, strategy pattern generation, and agent execution.
- It must not be used as benchmark evidence.
- Retry policy:
  - use a lower-concurrency config
  - keep the same frozen manifest
  - lower benchmark eval concurrency at launch time

## Retry Config

- Retry config:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/configs/config_openai_quality_stable.yaml`
- Key limits:
  - `document_processing.max_workers = 4`
  - `knowledge_graph_builder.max_workers = 4`
  - `narrative_graph_builder.max_workers = 4`
  - `strategy_memory.training_max_workers = 4`
- Benchmark script was also adjusted so offline / online training respects `strategy_memory.training_max_workers` instead of forcing `16`.

## Recording

- Progress source:
  - per-setting progress JSON under `reports/progress/`
- Final result source:
  - benchmark-level JSON and Markdown under `reports/`
- Runtime artifacts:
  - offline runtime library under `runtime/offline/`
  - online incremental runtime libraries under `runtime/online_incremental/`
