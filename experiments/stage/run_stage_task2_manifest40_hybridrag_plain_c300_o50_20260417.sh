#!/usr/bin/env bash
set -euo pipefail

source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver

RUN_NAME="stage_task2_manifest40_hybridrag_plain_c300_o50_r5_20260417"
RUN_ROOT="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/${RUN_NAME}"
WORKSPACE_ASSET_ROOT="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/assets/article_workspaces_hybridrag_plain_c300_o50"
MANIFEST_PATH="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/stage_task2_manifest40_newflow_r5_20260416/manifest.json"

mkdir -p "${RUN_ROOT}"
mkdir -p "${WORKSPACE_ASSET_ROOT}"

python experiments/stage/run_stage_task2_hybrid_rag_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path "${MANIFEST_PATH}" \
  --languages all \
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
