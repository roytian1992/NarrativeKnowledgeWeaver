# Reusable Experiment Assets Handoff

## Scope

This handoff records the reusable experiment assets currently stored under
`NarrativeWeaver`, so future sessions can restart STAGE / FiarytableQA runs
without rebuilding everything from scratch.

This document is asset-focused rather than paper-focused.

## Current State

- The most important reusable graph-backed assets are the `STAGE` formal
  40-movie workspaces under `experiments/stage/assets/article_workspaces`.
- The most important reusable retrieval-only assets are:
  - `STAGE` Hybrid RAG 40-movie workspaces under
    `experiments/stage/assets/article_workspaces_hybridrag_plain_c300_o50`
  - `FiarytableQA` Hybrid RAG 100-article workspaces under
    `experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain_c300_o50`
- `FiarytableQA` also has graph-backed agent workspaces for at least the key
  probe articles `a-fish` and `Snow-man` under
  `experiments/fiarytableqa/assets/article_workspaces`.
- There is a partially completed `FiarytableQA` 100-article agent rerun under
  `experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_tooltrim4w_20260418`,
  but it does not have a final `summary.json`, so it should be treated as a
  reusable intermediate state, not a finalized benchmark result.
- `QUALITY` in this repo currently has scripts and config, but no clean full
  benchmark asset directory that should be treated as the canonical restart
  point. The trusted `QUALITY` results still live outside this repo.

## Trusted Results

These results are included because they are tied to reusable asset roots.

### STAGE NarrativeWeaver Main Run

- run: `stage_task2_manifest40_newflow_r5_20260416`
- setting: `no_strategy_agent`
- movie count: `40`
- question count: `1241`
- overall: `0.4637`
- pass: `0.7575`
- avg latency: `46544.3 ms`

Main summary:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_newflow_r5_20260416/reports/summary.json`

Reusable graph-backed asset root:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces`

This root contains per-movie workspaces with:

- `knowledge_graph/`
- `narrative_graph/`
- `vector_store/`
- `sql/`
- `interactions/`
- `build_marker.json`

### STAGE Hybrid RAG Formal Baseline

- run: `stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417`
- setting: `hybrid_rag_plain`
- movie count: `40`
- question count: `1241`
- overall: `0.4611`
- pass: `0.4923`
- avg latency: `6296.03 ms`

Summary:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417/reports/summary.json`

Alignment note:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417/comparison_policy_20260419.md`

Reusable retrieval asset root:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces_hybridrag_plain_c300_o50`

This root contains per-movie retrieval workspaces with:

- `chunks.json`
- `vector_store/`
- `build_marker.json`

### FiarytableQA Hybrid RAG Full-100 Baseline

- run: `fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418`
- setting: `hybrid_rag_plain`
- article count: `100`
- question count: `4100`
- overall: `0.8406`
- pass: `0.8500`
- avg latency: `5818.56 ms`

Summary:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418/reports/summary.json`

Reusable retrieval asset root:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain_c300_o50`

This root contains per-article retrieval workspaces with:

- `chunks.json`
- `vector_store/`
- `build_marker.json`

## Untrusted Or Failed Results

### FiarytableQA Partial 100-Article Agent Rerun

Path:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_tooltrim4w_20260418`

Status:

- has `article_workspaces/`
- has many per-article `reports/*.json`
- has `progress.json`
- does **not** have a final `summary.json`

Usage rule:

- can be reused as an intermediate asset cache
- should **not** be cited as a finalized result

### FiarytableQA Single-Article / Small Probe Runs

Examples:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_afish_lean_recovery_20260419`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_snowman_focus3_queryrewrite3_20260419`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/tmp`

Usage rule:

- useful for tool/prompt debugging
- not suitable as the canonical restart point for a full benchmark rerun

### QUALITY Temporary Workdirs

Examples:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/tmp_subset_a_370f48d773951d4d_q5`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/tmp_debug_single_question`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/tmp_debug_process_question`

Usage rule:

- treat as ad hoc temp/debug state only
- do not use as the canonical `QUALITY` restart base

### STAGE Comparison Workspaces

Path:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/compare_runs/ch060a57_kg_cmp_20260416_v2`

Usage rule:

- useful for understanding graph size / relation-count changes on one movie
- not a benchmark restart point

## Important Paths

### Shared Config

- main config:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/configs/config_openai_quality_stable.yaml`

Important note:

- this config currently uses `Qwen3-235B`

### STAGE

- benchmark root:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0`
- converted scripts:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/converted_scripts`
- reusable graph-backed workspaces:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces`
- reusable hybrid retrieval workspaces:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces_hybridrag_plain_c300_o50`
- main formal run:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_newflow_r5_20260416`
- hybrid formal run:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417`
- removed-question alignment:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_newflow_r5_20260416/removed_question_alignment_20260417.json`

### FiarytableQA

- benchmark root:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/FiarytableQA`
- converted articles:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/converted_articles`
- graph-backed agent workspaces:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces`
- hybrid retrieval workspaces:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain_c300_o50`
- full-100 hybrid run:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418`
- partial 100-article agent rerun:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_tooltrim4w_20260418`

### QUALITY

- benchmark script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/run_quality_benchmark.py`
- alternate quality pipeline:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/quality/run_quality_two_graphstore_pipeline.py`

## Important Scripts

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_benchmark.py`
  - main STAGE graph-backed benchmark
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_hybrid_rag_benchmark.py`
  - STAGE hybrid RAG benchmark
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_manifest40_newflow_20260416.sh`
  - shell wrapper for the STAGE formal 40-movie graph-backed run
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/run_stage_task2_manifest40_hybridrag_plain_c300_o50_20260417.sh`
  - shell wrapper for the STAGE formal 40-movie hybrid run
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`
  - FiarytableQA graph-backed / agent benchmark
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytableqa_hybrid_rag_benchmark.py`
  - FiarytableQA hybrid RAG benchmark
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytableqa_manifest100_hybridrag_plain_c300_o50_20260418.sh`
  - shell wrapper for the FiarytableQA full-100 hybrid run

## What Was Tried

- A full graph-backed STAGE 40-movie run was executed and then sanitized to the
  current formal question files.
- A matching STAGE Hybrid RAG baseline was built on top of reusable retrieval
  workspaces using `chunk_size=300` and `chunk_overlap=50`.
- A full FiarytableQA 100-article Hybrid RAG baseline was built with reusable
  retrieval workspaces using the same chunking setup.
- FiarytableQA agent experiments then branched into many prompt/tool probes,
  especially around `a-fish` and `Snow-man`.
- A partial FiarytableQA 100-article agent rerun with trimmed tools was started
  and left behind reusable per-article workspaces and reports, but no final
  summary.
- QUALITY scripts were ported into this repo, but the trusted clean benchmark
  result lineage remained outside the repo, so local QUALITY temp dirs were not
  promoted to reusable asset roots.

## What Was Learned

- For STAGE graph-backed reruns, the key reusable unit is
  `experiments/stage/assets/article_workspaces/<movie_id>/`.
- For STAGE Hybrid RAG and FiarytableQA Hybrid RAG, the key reusable unit is
  the retrieval workspace root containing per-item `chunks.json` and
  `vector_store/`.
- To reuse assets, do **not** pass `--rebuild-workspaces`.
- To continue an interrupted or iterative run safely, prefer
  `--skip-existing-reports`.
- STAGE formal comparisons must stay aligned with the current
  `STAGE_v0/*/task_2_question_answering.csv`.
- FiarytableQA probe runs are useful for method debugging, but should not be
  mistaken for full benchmark restart anchors.

## Recommended Next Steps

1. Use the STAGE graph-backed workspace root as the default restart base for any
   future STAGE NarrativeWeaver rerun.
2. Use the STAGE and FiarytableQA Hybrid RAG workspace roots as the default
   restart bases for retrieval baselines.
3. If FiarytableQA agent full-set evaluation is resumed, start from the partial
   `fiarytableqa_no_strategy_100_tooltrim4w_20260418` run only after checking
   exactly how many article reports are already complete.
4. Do not try to bootstrap a formal QUALITY rerun from the local temp folders in
   this repo; re-import the trusted lineage or rerun cleanly.

## Command Template

### Reuse STAGE graph-backed workspaces

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay

python experiments/stage/run_stage_task2_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_newflow_r5_20260416/manifest.json \
  --workspace-asset-root /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces \
  --languages all \
  --repeats 5 \
  --eval-max-workers 8 \
  --build-max-workers 32 \
  --skip-existing-reports \
  --run-name stage_task2_rerun_from_reused_assets
```

### Reuse STAGE Hybrid RAG workspaces

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay

python experiments/stage/run_stage_task2_hybrid_rag_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417/manifest.json \
  --workspace-asset-root /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces_hybridrag_plain_c300_o50 \
  --repeats 5 \
  --eval-max-workers 8 \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --dense-top-k 8 \
  --bm25-top-k 8 \
  --final-top-k 8 \
  --skip-existing-reports \
  --run-name stage_task2_hybrid_rerun_from_reused_assets
```

### Reuse FiarytableQA Hybrid RAG workspaces

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay

python experiments/fiarytableqa/run_fiarytableqa_hybrid_rag_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418/manifest.json \
  --workspace-asset-root /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain_c300_o50 \
  --repeats 5 \
  --eval-max-workers 8 \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --dense-top-k 8 \
  --bm25-top-k 8 \
  --final-top-k 8 \
  --skip-existing-reports \
  --run-name fiarytableqa_hybrid_rerun_from_reused_assets
```
