from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy_training.online_strategy_training_runner import OnlineStrategyTrainingRunner
from core.utils.config import KAGConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train online retrieval strategy memory from questions only.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--csv", default="examples/datasets/we2_qa.csv")
    parser.add_argument("--dataset-name", default="we2_online")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--self-bootstrap-max-questions", type=int, default=1)
    parser.add_argument("--enable-sql-tools", action="store_true")
    args = parser.parse_args()

    cfg = KAGConfig.from_yaml(args.config)
    if args.workers is not None and int(args.workers) > 0:
        cfg.strategy_memory.training_max_workers = int(args.workers)

    runner = OnlineStrategyTrainingRunner(
        config=cfg,
        csv_path=args.csv,
        dataset_name=args.dataset_name,
        attempts_per_question=args.attempts,
        question_limit=args.limit,
        enable_sql_tools=args.enable_sql_tools,
        max_llm_calls_per_attempt=args.max_llm_calls,
        self_bootstrap_max_questions=args.self_bootstrap_max_questions,
    )
    try:
        result = runner.run()
    finally:
        runner.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
