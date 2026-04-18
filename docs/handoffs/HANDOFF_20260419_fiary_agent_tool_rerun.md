# Scope

This handoff covers the current active work in `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver`, specifically:

- FiarytableQA agent-side tool visibility and planner-loop tuning
- the active 2-article validation rerun on `Snow-man` and `a-fish`
- strict asset reuse with no KG / narrative rebuild

This handoff does not attempt to restate the older `NarrativeKnowledgeWeaver_langgraph` status. Current work should continue in `NarrativeWeaver`, not the old repo.

# Current State

- Active repo: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver`
- Dirty worktree is expected. Relevant modified files for this thread are:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/functions/tool_calls/composite_tools.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/task_specs/tool_metadata/en/graphdb_tools.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/task_specs/tool_metadata/zh/graphdb_tools.json`
- Stable baseline for the current Fiary validation is still:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_gapquery_20260418`
- Active experimental branch for this thread is:
  - hide `fact_timeline_resolution_search`
  - hide `narrative_hierarchical_search`
  - keep first-round backbone anchored by `bm25_search_docs` and `vdb_search_sentences`
  - strict asset reuse via `--reuse-existing-only`
- Current active run:
  - tmux session: `fiary_first2_reuse_notimeline_nonarr_20260419`
  - run dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419`
  - log: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/run.log`
- Current rerun status at handoff time:
  - `Snow-man` finished and report exists
  - `a-fish` is in progress
  - process is still alive
  - `progress.json` exists but is article-local; do not trust its `overall_progress=1.0` as run-global completion

Current Fiary benchmark code state:

- In `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`
  - `--reuse-existing-only` was added
  - when this flag is set, the run uses `load_existing_article_workspace_or_raise(...)` instead of build-or-prepare logic
  - Fiary `base_hidden_tool_names` now additionally hides:
    - `fact_timeline_resolution_search`
    - `narrative_hierarchical_search`
- In `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
  - current S4 runtime settings in effect for this thread are:
    - `max_tool_rounds_per_run = 3`
    - `first_round_max_tool_calls = 2`
    - `followup_round_max_tool_calls = 3`
    - `parallel_tool_workers = 4`
    - `bypass_runtime_tool_router = True`
  - round-1 backbone forcibly adds:
    - `bm25_search_docs`
    - `vdb_search_sentences`
    - `section_evidence_search`
    - plus question-type-specific extra tools if applicable
  - first-round disallowed lookup/follow-up tools include:
    - `vdb_get_docs_by_document_ids`
    - `lookup_titles_by_document_ids`
    - `lookup_document_ids_by_title`
    - `search_related_content`
    - `get_interactions_by_document_ids`
  - current follow-up planner guidance lives in `_planner_messages(...)` in this file and is the authoritative version

# Trusted Results

Trusted because the files exist and were read directly from disk.

Fiary 2-article baselines:

- Old best agent run on `a-fish`:
  - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_gapquery_20260418/reports/a-fish.json`
  - summary:
    - `overall_accuracy = 0.7250`
    - `pass_accuracy = 0.8750`
    - `pass_question_count = 14 / 16`
    - `avg_latency_ms = 18994`
- Old best agent run on `Snow-man`:
  - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_gapquery_20260418/reports/Snow-man.json`
  - summary:
    - `overall_accuracy = 0.6833`
    - `pass_accuracy = 0.8750`
    - `pass_question_count = 21 / 24`
    - `avg_latency_ms = 19303`
- Hybrid plain baseline on `a-fish`:
  - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418/reports/a-fish.json`
  - summary:
    - `overall_accuracy = 0.7125`
    - `pass_accuracy = 0.7500`
    - `pass_question_count = 12 / 16`
    - `avg_latency_ms = 5104.02`

Trusted partial result from the current rerun:

- Current rerun `Snow-man` report:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/reports/Snow-man.json`
  - summary:
    - `overall_accuracy = 0.8083`
    - `pass_accuracy = 0.8333`
    - `pass_question_count = 20 / 24`
    - `avg_latency_ms = 12256`
- This is already better than the old `Snow-man` run on overall accuracy and latency, but slightly lower on pass accuracy.

Asset reuse prerequisites were verified and are currently satisfied for both subset articles:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces/Snow-man`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces/a-fish`

Verified present in both workspaces:

- `build_marker.json`
- `knowledge_graph/graph_runtime_langgraph.pkl`
- `narrative_graph/global/episodes.json`
- `narrative_graph/global/storylines.json`
- `interactions/interaction_results.json`
- `sql/Interaction.db`

# Untrusted Or Failed Results

- Do not use this run as final until `a-fish.json`, `summary.json`, and final aggregate outputs are written:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419`
- Do not trust prompt-memory summaries from the long chat. The authoritative current planner prompt is only what is in:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
- Known regressed / contaminated Fiary run:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_gapquery_align2_descfix_20260418/reports/a-fish.json`
  - `a-fish` summary there was:
    - `overall_accuracy = 0.6500`
    - `pass_accuracy = 0.8125`
    - `avg_latency_ms = 11880`
  - This run is not a clean comparison target because the harmful high-level tools were still exposed and were observed in logs.

# Important Paths

Main repo:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver`

Main benchmark entrypoint:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`

Planner/runtime code:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/tool_router.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retriever_agent_langchain.py`

Composite tool definitions:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/functions/tool_calls/composite_tools.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/functions/tool_calls/graphdb_tools.py`

Tool metadata:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/task_specs/tool_metadata/en/graphdb_tools.json`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/task_specs/tool_metadata/zh/graphdb_tools.json`

Current active run:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/run.log`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/manifest.json`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/reports/progress.json`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/reports/Snow-man.json`

Subset input used by the current rerun:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/tmp/fiary_first2_subset/articles/Snow-man.json`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/tmp/fiary_first2_subset/articles/a-fish.json`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/tmp/fiary_first2_subset/questions/Snow-man.csv`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/tmp/fiary_first2_subset/questions/a-fish.csv`

Reusable article workspaces:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces/Snow-man`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces/a-fish`

Relevant docs already written in this repo:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/docs/graph_builder_speed_optimization_2026-04-15.md`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/docs/narrative_graph_p2_candidate_optimization_2026-04-15.md`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/docs/handoffs/2026-04-14_current_status_handoff.md`

# Important Scripts

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/run_fiarytable_benchmark.py`
  - Fiary benchmark driver
  - now supports `--reuse-existing-only`
  - owns Fiary hidden-tool policy for this thread
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/agent/retrieval/langgraph_runtime.py`
  - current planner-tools-evaluator-finalizer logic
  - first-round backbone tools
  - first-round disallowed tool list
  - follow-up round human prompt
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/core/functions/tool_calls/composite_tools.py`
  - tool definitions for `narrative_hierarchical_search` and `fact_timeline_resolution_search`
  - those tools are currently hidden for Fiary rerun, but this file still defines them

# What Was Tried

- Added hybrid plain RAG baseline for Fiary outside the agent path.
- Tested multiple agent prompt variants around follow-up planning.
- Recovered better behavior by returning to human-prompt-driven follow-up guidance instead of relying on system-prompt shaping.
- Forced round-1 evidence bundle to include `bm25_search_docs` and `vdb_search_sentences`.
- Reduced first-round breadth from earlier noisier settings to the current `first_round_max_tool_calls = 2`.
- Hid obviously harmful or debug-oriented tools in Fiary.
- After observing regressions on `a-fish`, specifically removed:
  - `fact_timeline_resolution_search`
  - `narrative_hierarchical_search`
- Launched a clean rerun that must reuse existing article workspaces rather than rebuild them.

# What Was Learned

- `a-fish` is highly sensitive to overly abstract / high-level tools. Both `fact_timeline_resolution_search` and `narrative_hierarchical_search` are plausible causes of degradation there.
- For Fiary open QA, a strong first-pass local-evidence bundle is important. `bm25_search_docs` and `vdb_search_sentences` are the most reliable anchors.
- Current rerun logs confirm that after hiding those two tools, the visible QA tools are now mainly:
  - `bm25_search_docs`
  - `vdb_search_sentences`
  - `section_evidence_search`
  - `retrieve_entity_by_name`
- `--reuse-existing-only` prevents rebuild from raw article text, but the run can still rewrite `graph_runtime_langgraph.pkl` during local graph / embedding cache sync. This is reuse, not a fresh graph build.
- Absence of `Chunking from:` and empty-graph initialization logs is the clean signal that no rebuild happened.
- `progress.json` is not a safe global completion signal for this run shape. Use report files plus process state instead.
- Current partial result suggests the tool trimming is promising on `Snow-man`:
  - old best `overall_accuracy = 0.6833`, current partial rerun `overall_accuracy = 0.8083`
  - old best `avg_latency_ms = 19303`, current partial rerun `avg_latency_ms = 12256`

# Recommended Next Steps

1. Let the active tmux run finish and do not restart it unless the process dies.
2. Once `a-fish.json` appears, read:
   - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/reports/a-fish.json`
   - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/reports/summary.json`
3. Compare `a-fish` against exactly these three baselines:
   - old best agent `0.7250 / 0.8750 / 18994 ms`
   - hybrid plain `0.7125 / 0.7500 / 5104.02 ms`
   - regressed align2-descfix `0.6500 / 0.8125 / 11880 ms`
4. If `a-fish` is still weak after this cleaner rerun, next likely ablation is `retrieve_entity_by_name`, but do not change code before reading the final result.
5. If a stricter no-write reuse mode is desired, add a second code path that loads existing workspace assets without persisting the runtime graph pickle.

# Command Template

Monitor the active run:

```bash
tail -f /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/fiarytableqa_first2_followup3_reuse_noft_nohier_20260419/run.log
```

Attach to tmux:

```bash
tmux attach -t fiary_first2_reuse_notimeline_nonarr_20260419
```

Rerun the same subset with strict reuse:

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver
python experiments/fiarytableqa/run_fiarytable_benchmark.py \
  --benchmark-root /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/tmp/fiary_first2_subset \
  --limit-articles 0 \
  --repeats 5 \
  --eval-max-workers 2 \
  --reuse-existing-only \
  --run-name fiarytableqa_first2_followup3_reuse_noft_nohier_20260419
```
