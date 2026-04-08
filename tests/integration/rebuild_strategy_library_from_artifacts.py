from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy_training.strategy_library_rebuilder import StrategyLibraryRebuilder
from core.utils.config import KAGConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild runtime strategy library from existing training artifacts.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-dir-name", default="rebuild_from_artifacts")
    parser.add_argument("--runtime-library-path", default=None)
    args = parser.parse_args()

    cfg = KAGConfig.from_yaml(args.config)
    rebuilder = StrategyLibraryRebuilder(
        config=cfg,
        training_root=args.training_root,
        output_dir_name=args.output_dir_name,
        runtime_library_path=args.runtime_library_path,
    )
    result = rebuilder.rebuild()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
