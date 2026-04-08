import argparse
import logging

from core import KAGConfig
from core.builder.community_graph_builder import CommunityGraphBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CommunityGraphBuilder only on the existing local runtime graph."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_openai.yaml",
        help="Path to KAG config YAML",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="",
        help="Unused compatibility argument. Community-only run reads from the existing runtime graph.",
    )
    parser.add_argument(
        "--clear_previous_community",
        action="store_true",
        help="Delete existing Community nodes and community assignments before rebuilding.",
    )
    parser.add_argument(
        "--clear_previous_narrative",
        action="store_true",
        help="Delete Episode/Storyline nodes before rebuilding communities.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = KAGConfig.from_yaml(args.config)
    config.global_.aggregation_mode = "community"
    config.global_config.aggregation_mode = "community"

    builder = CommunityGraphBuilder(config)
    try:
        result = builder.run(
            clear_previous_community=bool(args.clear_previous_community),
            clear_previous_narrative=bool(args.clear_previous_narrative),
        )
        print(result)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
