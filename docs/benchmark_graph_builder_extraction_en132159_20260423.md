# Graph Builder Extraction Benchmark (en132159, 2026-04-23)

## Setup

- workspace: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces/en132159a79f6040c88b523d3a87ec3224`
- stage: `graph_builder.extract_entity_and_relation(...)` only
- env: `screenplay`
- model endpoint: `http://localhost:8002/v1`
- model: `Qwen3-235B`
- workers: `4`
- legacy mode: `pipeline_mode=legacy`, `relation_extraction_mode=schema_direct`
- fast mode: `pipeline_mode=event_first_fast`, `relation_extraction_mode=open_then_ground`
- fast-mode short-scene skip threshold: `25` words
- raw outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_extraction_en132159_4workers_tmux_20260423/benchmark_summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_extraction_en132159_4workers_tmux_20260423/benchmark_report.md`

## Headline Result

Fast mode reduced extraction runtime from `1551.648s` to `876.079s`.

- time saved: `675.569s`
- speedup: `43.539%`

This is a real graph_builder-stage gain under the requested `4 workers` setting, not a single-chunk proxy.

## Count Comparison

| mode | docs | skipped_docs | entities | relations | avg_rel/doc | avg_rel/non_skipped_doc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 89 | 0 | 1080 | 1942 | 21.820 | 21.820 |
| fast | 89 | 13 | 1160 | 1325 | 14.888 | 17.434 |

Key deltas:

- entity count increased from `1080` to `1160` (`+7.4%`)
- relation count decreased from `1942` to `1325` (`-31.8%`)
- after removing skipped short scenes, relation density still dropped from `21.820` to `17.434` (`-20.1%`)
- fast mode skipped `13` ultra-short scenes and emitted section-only structure for them

Skipped docs:

- `scene_24_part_1`
- `scene_35_part_1`
- `scene_36_part_1`
- `scene_44_part_1`
- `scene_52_part_1`
- `scene_55_part_1`
- `scene_62_part_1`
- `scene_68_part_1`
- `scene_69_part_1`
- `scene_70_part_1`
- `scene_78_part_1`
- `scene_80_part_1`
- `scene_83_part_1`

## Relation Distribution

Top relation counts:

| relation_type | legacy | fast | delta |
| --- | ---: | ---: | ---: |
| performs | 451 | 422 | -29 |
| occurs_at | 413 | 269 | -144 |
| located_at | 297 | 201 | -96 |
| occurs_on | 209 | 35 | -174 |
| undergoes | 189 | 102 | -87 |
| experiences | 46 | 181 | +135 |
| participates_in | 25 | 52 | +27 |
| part_of | 80 | 17 | -63 |
| occurs_during | 36 | 29 | -7 |
| possesses | 39 | 6 | -33 |

Relations present in legacy but essentially missing in fast:

- `affinity_with`: `80 -> 0`
- `kinship_with`: `26 -> 0`
- `hostility_with`: `22 -> 0`
- `member_of`: `15 -> 0`
- `is_a`: `11 -> 0`

Interpretation:

- the fast pipeline preserves the high-volume event-core relation `performs` fairly well
- the biggest recall loss is on time/place/structure relations: `occurs_on`, `occurs_at`, `located_at`, `part_of`
- some event attachment mass shifted from `undergoes` to `experiences` and `participates_in`
- social and taxonomic relations are currently under-grounded in the fast path

## Entity Distribution

Top entity counts:

| entity_type | legacy | fast | delta |
| --- | ---: | ---: | ---: |
| Event | 468 | 475 | +7 |
| Character | 240 | 252 | +12 |
| Object | 134 | 187 | +53 |
| Location | 140 | 144 | +4 |
| Concept | 35 | 51 | +16 |
| TimePoint | 55 | 33 | -22 |
| Occasion | 8 | 18 | +10 |

Interpretation:

- the event-first flow is not losing entities overall
- `Object`, `Concept`, and `Occasion` extraction increased
- `TimePoint` extraction dropped, which aligns with the large `occurs_on` loss

## Document-Level Pattern

Largest relation-count drops were concentrated in a subset of heavier scenes:

- `scene_18_dup_2_part_1`: `69 -> 18`
- `scene_23_part_1`: `64 -> 22`
- `scene_1_part_1`: `59 -> 27`
- `scene_18_part_1`: `74 -> 43`
- `scene_2_part_1`: `60 -> 37`

There are also a few scenes where fast mode extracted more relations:

- `scene_81_part_1`: `36 -> 50`
- `scene_60_part_1`: `9 -> 21`
- `scene_42_part_1`: `1 -> 11`
- `scene_65_part_1`: `0 -> 8`

This suggests the fast pipeline is not simply weaker across the board. It is better on some locally dense event scenes, but currently misses too many schema-aligned time/place/social links.

## Takeaways

1. The speed improvement is already meaningful enough to justify keeping a fast mode in graph_builder.
2. The current fast path is entity-healthy but relation-recall-limited.
3. The missing relation mass is not random. It is concentrated in:
   - time grounding
   - location grounding
   - structural `part_of`
   - social/taxonomic relations
4. Short-scene skipping looks safe and should stay.

## Recommended Next Fixes

1. Strengthen the open relation extraction prompt with explicit target buckets instead of fully free extraction.
   - Ask the model to over-generate candidate relations for `time`, `place`, `event-participant`, `event-object`, `event-part-of`, and `social/taxonomic` buckets.
2. Add an entity-coverage fix after open relation grounding.
   - For non-skipped docs, check which `Character`, `Location`, `TimePoint`, and `Object` entities are not connected to any relation.
   - Run a cheap targeted repair that only tries to attach uncovered entities to nearby events.
3. Add a time/place-biased fix.
   - If a doc has `Event` plus `TimePoint` or `Location` but too few `occurs_on` / `occurs_at` / `located_at`, trigger a targeted grounding repair.
4. Broaden schema grounding for social relations.
   - Current open-to-schema grounding is effectively dropping `affinity_with`, `kinship_with`, `hostility_with`, `member_of`, and `is_a`.
   - That likely needs either richer candidate mapping guidance or a dedicated social-relation repair pass.
5. Consider keeping auto-built event relations as provisional edges before final pruning.
   - Right now the fast run reported `1471` auto relations but only `1325` final relations overall.
   - We should inspect where those edges are being discarded and whether some can survive grounding.

## Bottom Line

On this movie, the current fast graph_builder extraction mode is about `43.5%` faster than legacy under `4 workers`, but it pays for that with about `31.8%` fewer final relations.

My read is that the architecture direction is correct, but the open-relation prompt and post-grounding fix still need one more round focused on relation recall, especially for time/place/social coverage.
