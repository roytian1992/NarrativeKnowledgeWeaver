# Graph Builder Extraction Benchmark (en132159, fast_v2, 2026-04-23)

## Setup

- workspace: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces/en132159a79f6040c88b523d3a87ec3224`
- stage: `graph_builder.extract_entity_and_relation(...)`
- env python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python`
- endpoint: `http://localhost:8002/v1`
- model: `Qwen3-235B`
- workers: `4`
- legacy: `pipeline_mode=legacy`, `relation_extraction_mode=schema_direct`
- fast_v2: `pipeline_mode=event_first_fast`, `relation_extraction_mode=open_then_ground`

Compared against:

- legacy vs fast_v2 benchmark output:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_extraction_en132159_4workers_fastv2_20260423/benchmark_summary.json`
- previous fast_v1 benchmark output:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/tmp/benchmark_graph_builder_extraction_en132159_4workers_tmux_20260423/benchmark_summary.json`

## Headline Result

`fast_v2` successfully recovered a large amount of relation recall, but nearly eliminated the runtime advantage.

### legacy vs fast_v2

- legacy elapsed: `1636.034s`
- fast_v2 elapsed: `1568.290s`
- saved: `67.744s`
- saved pct: `4.141%`

### fast_v1 vs fast_v2

- fast_v1 elapsed: `876.079s`
- fast_v2 elapsed: `1568.290s`
- slower than fast_v1 by: `692.211s`
- slowdown pct vs fast_v1: `79.013%`

## Count Comparison

| mode | entities | relations | skipped_docs |
| --- | ---: | ---: | ---: |
| legacy | 1090 | 2004 | 0 |
| fast_v1 | 1160 | 1325 | 13 |
| fast_v2 | 1207 | 1816 | 13 |

Key changes:

- fast_v2 vs fast_v1 entity count: `+47`
- fast_v2 vs fast_v1 relation count: `+491`
- fast_v2 relation recall recovered to `90.6%` of legacy (`1816 / 2004`)
- fast_v1 relation recall was only `66.1%` of legacy (`1325 / 2004`)

## Relation Distribution

### legacy

- `performs`: `469`
- `occurs_at`: `425`
- `located_at`: `308`
- `occurs_on`: `207`
- `undergoes`: `170`
- `part_of`: `89`
- `affinity_with`: `73`
- `hostility_with`: `19`
- `kinship_with`: `27`
- `member_of`: `30`

### fast_v1

- `performs`: `422`
- `occurs_at`: `269`
- `located_at`: `201`
- `undergoes`: `102`
- `affinity_with`: `0`
- `hostility_with`: `0`
- `kinship_with`: `0`
- `member_of`: `0`

### fast_v2

- `performs`: `432`
- `occurs_at`: `311`
- `located_at`: `319`
- `undergoes`: `114`
- `affinity_with`: `58`
- `hostility_with`: `42`
- `kinship_with`: `17`
- `member_of`: `1`
- `possesses`: `154`
- `part_of`: `57`

## What Improved

The new fast_v2 changes clearly worked on recall:

- social relations came back:
  - `affinity_with`: `0 -> 58`
  - `hostility_with`: `0 -> 42`
  - `kinship_with`: `0 -> 17`
- core anchoring improved:
  - `occurs_at`: `269 -> 311`
  - `located_at`: `201 -> 319`
- action/event backbone improved:
  - `performs`: `422 -> 432`

The single-scene smoke on `scene_23_part_1` also confirmed that:

- `the wall` no longer gets misclassified as `Character`
- character relations are materially richer than fast_v1

## What Regressed

The current fast_v2 cost is too high.

Primary issue:

- the added grounding context plus open-relation bucketization plus coverage repair pushed runtime from `876s` to `1568s`
- that leaves only `4.1%` speed gain over legacy, which is not a meaningful fast mode anymore

Secondary issue:

- `possesses` jumped to `154`, which is much higher than legacy `35`
- `hostility_with` is now above legacy (`42` vs `19`)
- `located_at` is also slightly above legacy (`319` vs `308`)

This suggests fast_v2 is not only recovering missing edges. It is also overproducing in a few relation families.

## Interpretation

`fast_v2` proves that the architecture can recover the important relations.

That is useful, because it means the missing recall problem is solvable with:

- better entity grounding context
- bucketed open relation prompting
- coverage repair

But the current implementation is too expensive to keep as the default fast path.

The bottleneck is probably not one thing. It is the combination of:

1. richer grounding context
2. more aggressive open relation generation
3. an extra repair pass on many chunks

## Recommendation

Do not ship fast_v2 as-is.

Instead:

1. Keep the entity-side cleanup.
   - single grounding pass
   - direct typing for clear frame slots
   - grounding with access to source text
2. Keep bucketed open relation prompting.
3. Make coverage repair selective instead of unconditional.
   - only fire when Tier A gaps are detected
   - especially gate on missing social relations, missing `occurs_at`, and communicative events without `performs`
4. Add pruning for overproduced relation families.
   - especially `possesses`
   - likely also `hostility_with` when textual support is weak

## Bottom Line

`fast_v2` solved most of the recall problem but failed the speed goal.

It is the right direction for relation quality, but the current repair path is too expensive. The next iteration should keep the better entity grounding and better prompting, while making repair conditional so the fast path becomes meaningfully fast again.
