## NarrativeGraph P2 Candidate Optimization

This change keeps the existing graph schema and does not change:

- entity and relation schema validation
- graph-based important-node property extraction
- narrative aggregation from episode to storyline
- entity disambiguation and merge

What changed in `extract_episode_relations()`:

- `related_event_ids` and `related_occasion_ids` are treated as primary anchors
- `related_character_ids`, `related_location_ids`, and `related_time_ids` mainly reinforce or rerank existing pairs
- per-anchor bucket size is capped before pair expansion
- per-episode candidate degree is capped before LLM relation extraction
- candidate ranking now prefers weighted anchor evidence, not only raw shared-neighbor count

Main knobs in config:

- `episode_relation_primary_anchor_fields`
- `episode_relation_context_anchor_fields`
- `episode_relation_anchor_weights`
- `episode_relation_max_primary_bucket_size`
- `episode_relation_max_context_bucket_size`
- `episode_relation_context_requires_primary_pair`
- `episode_relation_topk_per_episode`
- `episode_relation_min_weighted_score`
