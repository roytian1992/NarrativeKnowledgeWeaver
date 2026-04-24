# Event-First Doc-Grounded Fast Benchmark (en3b3b6, 2026-04-24)

## Setup

- workspace: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces/en3b3b6db8683b4f509171b4b097837dbd`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python`
- endpoint: `http://localhost:8002/v1`
- model: `Qwen3-235B`
- workers: `8`
- external entity backend: `none`
- relation mode: `open_then_ground`
- baseline: `event_first_fast`
- new mode: `event_first_doc_grounded_fast`

## What Changed

The new mode keeps per-chunk Event/Occasion frame extraction unchanged, but moves two grounding steps from chunk-level to document-level:

1. Open entity grounding is executed once per document.
2. Relation-brought entity grounding is executed once per document.

Open relation extraction and coverage repair remain chunk-level.

## Input Shape Constraint

For this movie workspace:

- total documents: `66`
- skipped short documents: `7`
- chunk distribution: `{1: 19, 2: 47}`
- average chunks per document: `1.71`

This sharply limits the maximum benefit of document-level grounding, because most documents only contain one or two chunks.

## Summary

### Baseline `event_first_fast`

- summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_fast_only_en3b3b6_8workers_v9_keyword_20260424/benchmark_summary.json`
- elapsed: `951.851s`
- entities: `1275`
- relations: `1900`
- auto relations: `2059`
- open relations after fix: `194`
- coverage repair triggered chunks: `28`

### New `event_first_doc_grounded_fast`

- summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_fast_only_en3b3b6_8workers_v14_docground_20260424/benchmark_summary.json`
- elapsed: `929.224s`
- entities: `1281`
- relations: `1868`
- auto relations: `2059`
- open relations after fix: `154`
- coverage repair triggered chunks: `30`

## Delta

- elapsed: `-22.627s` (`-2.38%`)
- entities: `+6`
- relations: `-32`
- open relations after fix: `-40`

## Stage-Time Comparison

Aggregated over the `59` non-skipped documents:

### Baseline

- frame extraction: `2881.599s`
- open entity extraction: `406.859s`
- entity grounding: `435.468s`
- open relation extraction: `1863.276s`
- relation-brought entity grounding: `110.970s`
- relation grounding: `242.748s`
- coverage repair: `953.866s`

### New

- frame extraction: `2892.573s`
- open entity extraction: `418.362s`
- entity grounding: `432.231s`
- open relation extraction: `1917.863s`
- relation-brought entity grounding: `118.957s`
- relation grounding: `244.720s`
- coverage repair: `871.757s`

## Reading

This change is valid but not sufficient.

- The document-level grounding refactor does not break the pipeline and preserves overall graph volume reasonably well.
- The total speedup is small because the workload is not chunk-heavy enough per document.
- The largest remaining bottlenecks are still `frame_extraction`, `open_relation_extraction`, and `coverage_repair`.
- The next meaningful speedup likely needs to come from reducing relation extraction/repair work, not from further grinding on entity grounding granularity.

## Code Snapshot

- commit: `2627f7c`
- message: `Add doc-grounded fast extraction mode`
