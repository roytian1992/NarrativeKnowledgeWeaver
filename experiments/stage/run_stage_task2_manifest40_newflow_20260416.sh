#!/usr/bin/env bash
set -euo pipefail

source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver

RUN_NAME="stage_task2_manifest40_newflow_r5_20260416"
RUN_ROOT="/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeWeaver/experiments/stage/runs/${RUN_NAME}"
mkdir -p "${RUN_ROOT}"

python experiments/stage/run_stage_task2_benchmark.py \
  --config configs/config_openai_quality_stable.yaml \
  --manifest-path /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/stage/runs/stage_task2_en30_zh10_r5_w32_20260415/manifest.json \
  --languages all \
  --repeats 5 \
  --eval-max-workers 8 \
  --build-max-workers 32 \
  --rebuild-workspaces \
  --skip-existing-reports \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${RUN_ROOT}/run.log"
