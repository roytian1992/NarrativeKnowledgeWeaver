## Graph Builder Speed Optimization

This round expands optimization beyond narrative relation P2 and keeps these core properties unchanged:

- final graph still follows the existing schema
- important-node property extraction is preserved
- narrative aggregation is preserved
- entity disambiguation and merge are preserved

### Knowledge graph document extraction

- short neighboring chunks are packed before `run_document()`
- long chunks still stay isolated
- this reduces document-sequential LLM rounds without changing extraction schema

Main knobs:

- `knowledge_graph_builder.extraction_pack_short_chunk_word_threshold`
- `knowledge_graph_builder.extraction_pack_max_words`

### Property extraction

- repeated edge descriptions are deduplicated
- edge-description count per node is capped
- total context words per node are capped
- node selection by centrality is unchanged

Main knobs:

- `knowledge_graph_builder.property_context_max_edge_descriptions`
- `knowledge_graph_builder.property_context_max_total_words`
- `knowledge_graph_builder.property_context_dedupe_descriptions`

### Interaction extraction

- documents with too few eligible entity candidates are skipped early
- this avoids unnecessary LLM calls on documents that cannot produce valid interaction pairs

Main knob:

- `knowledge_graph_builder.interaction_min_entity_candidates`

### NarrativeGraph P2

See also:

- `docs/narrative_graph_p2_candidate_optimization_2026-04-15.md`
