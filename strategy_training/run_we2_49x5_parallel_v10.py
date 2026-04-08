from core.strategy_training.strategy_training_runner import StrategyMemoryTrainingRunner
from core.utils.config import KAGConfig
import json

cfg = KAGConfig.from_yaml('configs/config_openai.yaml')
cfg.strategy_memory.training_max_workers = 24
runner = StrategyMemoryTrainingRunner(
    config=cfg,
    csv_path='examples/datasets/we2_qa.csv',
    dataset_name='we2_49x5_parallel_v10',
    attempts_per_question=5,
    question_limit=None,
    answer_temperature=None,
    enable_sql_tools=False,
)
try:
    result = runner.run(reset_runtime_library=True)
    print('\n=== FINAL RESULT ===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    runner.close()
