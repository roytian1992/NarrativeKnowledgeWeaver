# QA Native TopK Baseline - 2026-04-28

## Status

Keep this version as the current preferred QA baseline for `en0c08ce1c06774785b5d73d9effd69e6b`.

The key decision is:

- Keep Qwen native tool routing enabled by default.
- Do not globally force the LLM router for QA.
- Keep `top_k_by_centrality` visible and strongly described in tool metadata.
- Use `top_k_by_centrality` for central-character ranking questions through native tool selection.

## Result

Run directory:

`tmp/qa_en0c08_native_topk150_qwen8001_20260428_053415`

Command:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python tmp/run_fast_graph_stage_task2_qa_en0c08.py \
  --config tmp/config_fast_benchmark_en_8001_localwindow.yaml \
  --workspace tmp/manifest40_en0c08ce1c06774785b5d73d9effd69e6b_full_fast_v13_localwindow_qwen8001_20260427 \
  --out-dir tmp/qa_en0c08_native_topk150_qwen8001_20260428_053415 \
  --repeats 5 \
  --workers 16 \
  --setting en0c08_native_topk150 \
  --language en
```

Summary:

- `attempt_count`: 150
- `overall_accuracy`: 0.6867
- `pass_accuracy@5`: 0.8667
- `pass_question_count`: 26/30
- `avg_latency_ms`: 100568.51
- `tool_call_avg`: 2.887

Failed pass questions:

- `q3`
- `q9`
- `q11`
- `q16`

Important improvement:

- `q19` central-character ranking changed from `0/5` to `5/5`.
- All `q19` attempts used `top_k_by_centrality`.
- Correct answer: `Clarice Starling, Jack Crawford, Hannibal Lecter`.

## Relevant Code State

Important files touched for this baseline:

- `core/functions/tool_calls/graphdb_tools.py`
- `core/agent/retriever_agent_qwen.py`
- `core/agent/retrieval/tool_router.py`
- `core/agent/retrieval/llm_router.py`
- `core/agent/retrieval/tool_capability_profile.py`
- `core/agent/retrieval/tool_family_registry.py`
- `task_specs/tool_metadata/en/graphdb_tools.json`
- `task_specs/tool_metadata/zh/graphdb_tools.json`

Notes:

- `qwen_native_tool_routing_only` default should remain enabled unless explicitly testing router behavior.
- `top_k_by_centrality` should remain visible and described as the deterministic graph ranking tool for centrality questions.
- Full router mode previously fixed `q19` but reduced total `pass@5` to 0.8000, so it should not be the default QA path.
