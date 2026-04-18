#!/usr/bin/env bash
set -euo pipefail

source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver

RUN_NAME="fiarytableqa_manifest100_hybridrag_plain_c300_o50_r5_20260418"
RUN_ROOT="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/runs/${RUN_NAME}"
WORKSPACE_ASSET_ROOT="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/fiarytableqa/assets/article_workspaces_hybridrag_plain_c300_o50"
MANIFEST_PATH="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/manifest.json"

mkdir -p "${RUN_ROOT}"
mkdir -p "${WORKSPACE_ASSET_ROOT}"

python experiments/fiarytableqa/run_fiarytableqa_hybrid_rag_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path "${MANIFEST_PATH}" \
  --repeats 5 \
  --eval-max-workers 8 \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --dense-top-k 8 \
  --bm25-top-k 8 \
  --final-top-k 8 \
  --workspace-asset-root "${WORKSPACE_ASSET_ROOT}" \
  --skip-existing-reports \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${RUN_ROOT}/run.log"
