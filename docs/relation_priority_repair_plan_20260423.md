# Relation Priority Repair Plan for Fast KG Extraction

## Goal

Keep the current fast pipeline speed advantage, but recover the relation categories that materially affect screenplay knowledge graph usefulness.

Target pipeline:

1. `event_occasion_frame_extraction`
2. `open_entity_extraction`
3. `entity_schema_grounding`
4. `open_relation_extraction`
5. `schema_relation_grounding`
6. targeted relation repair
7. existing dedup / validation / drop flow

The repair stage should be selective and schema-aware. It should not try to restore all legacy relations.

## Priority Tiers

### Tier A: Must Recover

These relations directly affect narrative querying, character reasoning, and event navigation.

- `hostility_with`
- `affinity_with`
- `kinship_with`
- `member_of`
- `performs` for dialogue / decision / threat / request / promise / instruction events
- `occurs_at` for core events
- `occurs_on` when grounded to a meaningful `TimePoint`
- `located_at` for main characters / key objects in scene-defining events

Why these matter:

- social relations carry stable character structure
- dialogue-driven `performs` carries plot progression
- `occurs_at` / `occurs_on` are needed for scene/event retrieval and chronology
- `located_at` helps attach agents, props, and events into local spatial context

### Tier B: Nice To Have

Useful, but not worth expensive repair by default.

- `participates_in`
- `undergoes`
- `experiences`
- `part_of` when it reflects meaningful nested setting structure
- `initiates`
- `occurs_during`

These can often be recovered indirectly from Tier A repairs or existing auto-relations.

### Tier C: Safe To Drop

Low information density or often redundant in screenplay extraction.

- `occurs_on` grounded to vague phrases like `that moment`, `moments later`
- broad environmental `located_at` for incidental props
- mechanical `part_of` chains like `street -> alley`, `office -> warehouse`, unless required by a downstream consumer
- weak `is_a` edges that mostly restate descriptions

## Main Insight from Manual Review

The fast pipeline is not mainly losing random edges. It is losing relation mass in three concentrated areas:

1. social relations are under-produced or dropped during grounding
2. dialogue and intention events are under-covered by open relation extraction
3. time/place anchoring is inconsistently attached to otherwise valid events

This means a generic “extract more relations” fix is not the right move. The repair should explicitly target these three buckets.

## Proposed Landing Changes

## 1. Change Open Relation Prompt From Free Recall To Bucketed Recall

Current problem:

- the prompt says “high recall”, but does not force the model to sweep the high-value buckets
- as a result, the model over-focuses on visible physical events and under-focuses on social / dialogue links

Prompt change:

- keep extraction open-form
- add a required internal sweep over relation buckets
- ask the model to over-generate candidates for high-value buckets first

Recommended bucket order:

1. character-character social relation
2. character-event action / speech / decision / instruction / threat relation
3. event-location relation
4. event-time relation
5. character/object-location relation
6. event-object / event-patient relation
7. organization-membership relation

Prompt rule:

- if a bucket is relevant to the text, try to emit at least one candidate before moving on
- do not require schema labels, only open relation phrases
- if the evidence is weak, skip it rather than hallucinating

Why this helps:

- it preserves the open-extraction philosophy
- it pushes recall toward the categories that actually matter

## 2. Add Lightweight Post-Grounding Coverage Repair

New stage:

- run after `schema_relation_grounding`
- operate only on non-skipped docs/chunks
- only fire if Tier A coverage looks insufficient

Coverage checks:

- any `Character` with zero incident relations
- any `TimePoint` with zero attached `occurs_on`
- any `Location` with zero attached `occurs_at` or `located_at`
- any paired `Character` mentions in strongly emotional / confrontational / familial dialogue without social relation edges
- any communicative `Event` with no `performs`

Repair input should be small:

- chunk text
- grounded entity list
- existing grounded relations
- uncovered target entities
- requested relation families

Repair output:

- grounded schema relations only
- allow `drop`
- no free-form explanation

This repair should be one extra LLM call per chunk at most, and only for chunks that fail coverage heuristics.

## 3. Add A Social-Relation Focused Repair Pass

Trigger when:

- chunk contains 2+ `Character` entities
- text contains strong interaction markers
- current relation set has zero social edges

Strong markers:

- threat / insult / argument / apology / family / sibling / dating / attraction / intimidation / loyalty

Expected schema targets:

- `hostility_with`
- `affinity_with`
- `kinship_with`
- `member_of`

Important rule:

- do not try to infer hidden psychology
- only emit social edges when the text directly supports them

This is important because current fast mode is dropping exactly these edges.

## 4. Add A Time/Place Anchor Repair Pass

Trigger when:

- chunk has `Event` plus `Location`, but too few `occurs_at`
- chunk has `Event` plus `TimePoint`, but zero meaningful `occurs_on`

Repair goal:

- attach only core events, not every micro-event
- prefer scene-defining events and dialogue/action pivots

Additional rule:

- drop vague time anchors like `that moment` unless the downstream graph explicitly needs dense temporal scaffolding

Net effect:

- recover useful anchors
- avoid the legacy behavior of overproducing low-value `occurs_on`

## 5. Preserve Auto-Built Event Relations More Carefully

Current signal:

- fast benchmark showed many auto relations are created, but the final grounded set is still much smaller than legacy

Recommendation:

- treat auto-built relations as provisional Tier B defaults
- do not let later grounding/drop logic wipe them out unless there is a direct contradiction or duplication
- especially preserve:
  - `performs`
  - `occurs_at`
  - meaningful `occurs_on`
  - `located_at`

This is a cheap way to hold on to event-frame structure without another expensive extraction call.

## Concrete Heuristics

Use these heuristics before deciding whether to call the repair model.

### Social Repair Trigger

Fire if all are true:

- `Character` count >= 2
- text contains dialogue or interaction markers
- count of `affinity_with + hostility_with + kinship_with + member_of` == 0

### Communicative Event Repair Trigger

Fire if both are true:

- chunk has communicative events from frame extraction
- those events have no `performs`

### Location Repair Trigger

Fire if all are true:

- `Location` count >= 1
- `Event` count >= 2
- `occurs_at` density is below threshold

Suggested threshold:

- `occurs_at_count < min(2, event_count // 2)`

### Time Repair Trigger

Fire if all are true:

- `TimePoint` count >= 1
- there is no meaningful `occurs_on`
- timepoint text is not in `{that moment, moments later, later, now}`

## Suggested Implementation Point

Best landing point is inside [event_first_extractor.py](/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver_DEV/core/builder/event_first_extractor.py), right after:

- auto-relations are created
- open relations are grounded/fixed

Recommended shape:

1. build `combined_rel_map`
2. run local coverage analysis
3. if coverage gap is Tier A relevant:
   - call `targeted_relation_repair(...)`
4. merge repaired relations
5. dedup once

This keeps the legacy path untouched and limits changes to fast mode only.

## Minimal First Version

If we want the smallest high-ROI patch first, do only this:

1. revise `extract_open_relations.yaml` to add bucketed recall instructions
2. add one targeted repair call for:
   - social relations
   - missing `occurs_at`
   - missing meaningful `occurs_on`
   - communicative events with no `performs`
3. keep all other fix logic unchanged

This should recover most of the important loss without destroying the current runtime win.

## Success Criteria For The Next Benchmark

The next fast benchmark should aim for:

- keep at least `30%+` runtime savings versus legacy
- recover most of the missing Tier A relations
- specifically:
  - social relation count no longer near zero
  - `occurs_at` gap significantly reduced
  - `performs` on dialogue-heavy scenes close to legacy
  - `occurs_on` stays selective rather than returning to legacy overproduction

## Recommendation

Do not optimize for total relation count.

Optimize for:

1. Tier A relation recall
2. low extra-call overhead
3. preserving the current fast pipeline shape

The right target is not “match legacy edge count”. The right target is “match legacy on the relations that matter, while staying materially faster”.
