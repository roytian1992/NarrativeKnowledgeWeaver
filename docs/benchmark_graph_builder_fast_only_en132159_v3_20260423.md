# Fast-Only Benchmark (en132159, v3, 2026-04-23)

## Setup

- workspace: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces/en132159a79f6040c88b523d3a87ec3224`
- mode: `pipeline_mode=event_first_fast`
- relation mode: `open_then_ground`
- workers: `4`
- model: `Qwen3-235B`
- endpoint: `http://localhost:8002/v1`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python`

## Summary

- elapsed: `1428.525s`
- docs: `89`
- skipped_docs: `13`
- entities: `1224`
- relations: `2030`
- auto relations: `1603`
- open relations after fix: `602`
- coverage repair triggered chunks: `33`

## Delta vs Previous Fast Runs

### vs fast_v2

- elapsed: `-139.765s`
- entities: `+17`
- relations: `+214`

### vs fast_v1

- elapsed: `+552.446s`
- entities: `+64`
- relations: `+705`

## Top Relation Types

- `located_at`: `458`
- `performs`: `430`
- `occurs_at`: `317`
- `possesses`: `183`
- `experiences`: `168`
- `undergoes`: `120`
- `affinity_with`: `79`
- `hostility_with`: `70`

## Notes

- This run used schema-derived relation samples as few-shot support.
- Fast coverage repair is now conditional, not unconditional.
- Fast-path relation dedup is currently disabled by design for recall measurement.

## Reading

This version moved in the right direction on both speed and recall relative to `fast_v2`, but relation counts are now above prior legacy counts because dedup is off and some relation families are overproduced.
