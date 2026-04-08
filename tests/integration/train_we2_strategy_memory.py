from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy_training.strategy_training_runner import StrategyMemoryTrainingRunner
from core.utils.config import KAGConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train retrieval strategy memory from WE2 QA pairs.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--csv", default="examples/datasets/we2_qa.csv")
    parser.add_argument("--dataset-name", default="we2")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--enable-sql-tools", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = KAGConfig.from_yaml(args.config)
    if args.workers is not None and int(args.workers) > 0:
        cfg.strategy_memory.training_max_workers = int(args.workers)
    runner = StrategyMemoryTrainingRunner(
        config=cfg,
        csv_path=args.csv,
        dataset_name=args.dataset_name,
        attempts_per_question=args.attempts,
        question_limit=args.limit,
        answer_temperature=args.temperature,
        enable_sql_tools=args.enable_sql_tools,
    )
    try:
        result = runner.run()
    finally:
        runner.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
