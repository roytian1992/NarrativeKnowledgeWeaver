# Quality Online Experiment Handoff

Last updated: 2026-03-18 17:01 Asia/Shanghai

## 1. What We Discussed

This thread focused on debugging and improving the `QUALITY` benchmark pipeline, especially the relation between:

- knowledge extraction / article workspace building
- online strategy evaluation / online training
- template matching / routing behavior
- multi-choice QA evaluation behavior

The most important correction is conceptual:

- Knowledge extraction is a separate frozen phase.
- Online evaluation/training must reuse an already-built article workspace.
- Online testing must not silently rebuild article chunking/entity extraction/narrative extraction.

The user explicitly called out that previous runs were wrong because article preparation and online evaluation were mixed together. This is now treated as a hard constraint.

## 2. Important User Requirements

Please preserve these assumptions in follow-up work:

- Do not silently rebuild article workspace during online testing.
- Reuse the existing extracted workspace for article-level experiments.
- If the workspace is missing or incomplete, fail loudly instead of falling back to rebuild.
- For this benchmark, the agent should be able to run in a "must use tools before answering" mode.
- Template usage should be conservative: better to miss than to force a weak match.
- For multiple-choice evaluation, the prompt should expose `A/B/C/D`, and prediction extraction can be rule-based from final answer text.
- Online strategy matching must not use GT-dependent signals. In particular, `success_rate` should not participate in online similarity/routing decisions if that leaks ground truth.
- The user wants better progress visibility and incremental result persistence.

## 3. Main Problems Identified Earlier

### 3.1 Wrong experiment boundary

Earlier online runs were re-triggering article build steps such as:

- chunking / pre-splitting
- document extraction
- entity/relation graph build

That was wrong for the current experiment design.

### 3.2 Document entity logic for long-text splitting

The user found that `doc_entities.json` had only one `Document` entity, while pre-splitting had already generated new titles/segments. The expectation is:

- the derived split titles/sections should become `Document` entities
- not only the original single article title

This issue was discussed as a logic problem in the document segmentation / document entity mapping stage.

### 3.3 Missing or sparse storyline / episode relations

The user observed:

- missing `storyline`
- weak or missing episode-to-episode relations

At that point we decided not to prioritize `storyline` first, and instead fix segmentation/extraction first.

### 3.4 QUALITY benchmark evaluation mismatch

We discussed that:

- training can use `answer_text`
- but test-time QA is multiple-choice, so prompts should include options
- answer judging should consider the multiple-choice format

There were also bugs around:

- empty `predicted_choice`
- hallucinated doc names like `doc_1999_year_in_review`
- formatting fragility in answer parsing

The intended direction became:

- extract `A/B/C/D` from final answer text using rule-based parsing
- enforce retrieval/tool use before final answer

### 3.5 Online strategy quality unexpectedly below no-strategy baseline

This triggered deeper debugging of:

- template matching threshold behavior
- whether low-confidence templates were still used
- whether tool outputs were noisy / bilingual
- whether online matching was contaminated by GT-derived fields
- whether routing/template accumulation/clustering was too heavy or too permissive

## 4. Main Code Changes Already Made

### 4.1 Hard separation between article workspace build and online-only evaluation

File:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/run_quality_benchmark.py`

Added logic:

- `load_existing_article_workspace_or_raise(...)`

Behavior:

- accepts a provided `workspace_dir`
- requires `workspace_dir/build_marker.json`
- clears current graph state
- reloads the existing article workspace into the local runtime graph
- raises if the workspace is missing/incomplete
- never silently calls article rebuild in reuse mode

This was the most important fix in this thread.

### 4.2 Added dedicated online-only runner from existing workspace

File:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/run_online_setting_from_workspace.py`

Purpose:

- run exactly one online setting from an already-built article workspace
- avoid re-running extraction/build steps

Used args include:

- `--run-name`
- `--article-name`
- `--workspace-dir`
- `--setting-name`
- `--repeats`
- `--eval-max-workers`
- `--online-warmup-questions`
- `--online-batch-size`
- `--self-bootstrap-max-questions`
- `--subagent`

### 4.3 Earlier runtime/training-path changes that were already in place

Files mentioned during this thread:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/core/strategy_training/strategy_cluster_manager.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/core/strategy_training/online_strategy_training_runner.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/core/strategy_training/strategy_training_runner.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/run_quality_benchmark.py`

Main directions discussed/implemented in that phase:

- lighter incremental clustering
- batch-level template grouping
- defer heavy final distillation to consolidation
- required-tool-use behavior
- conservative template use

## 5. Other Topics We Discussed

### 5.1 Thresholds and template selection

The user asked about:

- `match_min_score`
- `single_agent_min_selection_score`
- `subagent_min_selection_score`

Discussion outcome:

- the latter two are conceptually similar and could likely be unified later
- conservative selection is preferred
- if there is no sufficiently good template, the system should simply use none
- for subagent mode, a reasonable behavior is "matched templates + one no-template execution"

### 5.2 LLM routing hint

The user proposed that lightweight LLM selection could be folded into `routing_hint` instead of a separate heavy routing stage. We did not fully implement a new selector in this thread, but this is a valid future direction.

### 5.3 Tool output language / verbosity

The user pointed out that tool outputs were showing Chinese text such as:

- `搜索到以下实体`

Requested direction:

- keep benchmark/tool outputs in English
- remove unnecessary lead-in phrases
- return results directly

### 5.4 Narrative graph pair threshold fallback

For episode linkage / narrative graph builder, the user requested a dynamic floor mechanism:

- default similarity threshold remains around `0.55`
- if too few pairs are found, progressively reduce threshold by `0.05`
- guarantee at least a configured minimum number of candidate pairs, e.g. `10`

This was discussed as a better fallback than a single static threshold.

### 5.5 Experiment UX expectations

The user requested:

- save results immediately after each setting finishes
- better progress display
- more concurrency inside a batch because questions are independent

These are still important pending UX/perf follow-ups.

## 6. Current Online Experiment We Are Running

This is the current active run:

- session id: `24679`
- setting: `online_strategy_agent`
- article: `a_09e1543478c34e8e`
- repeats: `5`
- total eval attempts expected: `70`

Command:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -u \
  /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/run_online_setting_from_workspace.py \
  --run-name quality_online_strategy_agent_a09_reusefix_20260318_1640 \
  --article-name a_09e1543478c34e8e \
  --workspace-dir /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/runs/quality_train10_test50_onlinefix_20260317/article_workspaces/a_09e1543478c34e8e \
  --setting-name online_strategy_agent \
  --repeats 5 \
  --eval-max-workers 4 \
  --online-warmup-questions 5 \
  --online-batch-size 3 \
  --self-bootstrap-max-questions 1
```

### 6.1 Existing frozen workspace being reused

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/runs/quality_train10_test50_onlinefix_20260317/article_workspaces/a_09e1543478c34e8e`

### 6.2 Current run root

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/runs/quality_online_strategy_agent_a09_reusefix_20260318_1640`

### 6.3 Validation already confirmed

This corrected run does **not** re-run article build steps. We already verified there were no logs like:

- chunking documents
- extracting documents

Instead it reloads the existing workspace and then enters evaluation/training.

## 7. Current Status Snapshot

As of approximately `2026-03-18 17:00:54`:

Setting progress file:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/runs/quality_online_strategy_agent_a09_reusefix_20260318_1640/reports/progress/online_strategy_agent/a_09e1543478c34e8e/progress.json`

Reported values:

- `repeat_index = 1`
- `batch_index = 8 / 8`
- `evaluated_attempts_done = 11 / 70`
- `overall_progress = 0.1571`

Latest batch progress file:

- dataset: `a_09e1543478c34e8e_online_strategy_agent_r0_batch_008`
- status: `running`
- `completed_attempts = 15 / 15`
- `completed_questions = 3 / 3`
- `updated_at = 2026-03-18 17:00:53`

Important note:

- The JSON progress files can lag or be internally inconsistent.
- In the latest snapshot, the batch file still says `status = running` even though `15/15` attempts are complete.
- Therefore, liveness should be judged primarily from the live terminal output in session `24679`, not only from `progress.json`.

No final result file exists yet:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver/experiments/quality/runs/quality_online_strategy_agent_a09_reusefix_20260318_1640/reports/single_online_setting_result.json`

## 8. Important Interpretation Notes

### 8.1 What `15/15` means

The user asked what `15/15` means.

Meaning:

- current batch has `3` questions
- each question has `5` attempts / passes
- so total attempts in that batch = `3 * 5 = 15`

So `15/15` is **batch-local**, not global experiment completion.

### 8.2 Online training meaning in this project

The user explicitly clarified the intended meaning:

- online = test the agent while updating the strategy memory during the same run

This should remain the conceptual definition for future explanations and diagrams.

## 9. Known Pending Work

These items were requested or motivated in the thread and are still worth doing:

- save each setting result immediately when that setting finishes
- improve progress display, ideally with per-repeat / per-batch / per-question visibility
- improve concurrency inside a batch
- make progress persistence more reliable and real-time
- audit online strategy matching so no GT-derived signal leaks into similarity/routing
- keep tool output English-only and concise
- continue tuning thresholding / conservative template routing
- revisit clustering path for lightweight parallel candidate grouping if needed
- later revisit document segmentation to make split titles become `Document` entities
- later revisit storyline / episode relations after segmentation/extraction correctness is stable

## 10. Recommended Next Steps For a New Session

If a new session takes over, the most useful order is:

1. Monitor session `24679` until the current `online_strategy_agent` run finishes or clearly stalls.
2. If it stalls, inspect:
   - live terminal output from session `24679`
   - setting progress JSON
   - latest batch progress JSON
   - whether final result JSON was produced
3. Do **not** rebuild article workspace for this run.
4. Once the run finishes, summarize:
   - accuracy
   - pass accuracy if available
   - correct / total
   - average latency
   - any abnormal batches
5. Then implement the next UX/perf items:
   - save-per-setting
   - better progress display
   - batch concurrency

## 11. Short Version

If someone only reads one paragraph:

We discovered that online QUALITY experiments were wrongly rebuilding article knowledge extraction. That boundary has now been fixed by adding strict workspace-reuse logic and a dedicated online-only runner. The active experiment is `online_strategy_agent` on article `a_09e1543478c34e8e`, reusing the frozen workspace, in live session `24679`. It is still running and appears alive, though JSON progress updates are somewhat stale/inconsistent. Future work must preserve the rule that online testing never silently rebuilds extraction, and should next improve result persistence, progress visibility, concurrency, and conservative template routing.
