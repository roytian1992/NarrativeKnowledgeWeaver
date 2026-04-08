import argparse
import logging

from core import KAGConfig
from core.builder.community_graph_builder import CommunityGraphBuilder
from core.builder.graph_builder import KnowledgeGraphBuilder
from core.builder.narrative_graph_builder import NarrativeGraphBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the base KnowledgeGraphBuilder, then execute the configured aggregation pipeline."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_openai.yaml",
        help="Path to KAG config YAML",
    )
    return parser.parse_args()


def run_base_graph_pipeline(config: KAGConfig, *, json_file_path: str) -> None:
    builder = KnowledgeGraphBuilder(
        config,
        use_memory=True,
    )
    builder.prepare_chunks(json_file_path=json_file_path)
    builder.extract_entity_and_relation(
        retries=3,
        concurrency=64,
        per_task_timeout=2400,
        reset_outputs=True,
    )
    builder.run_extraction_refinement()
    builder.build_entity_and_relation_basic_info()
    builder.postprocess_and_save()
    builder.extract_properties()
    builder.extract_interactions()
    builder.store_interactions_to_sql()
    builder.build_doc_entities()
    builder.load_json_to_graph_store()


def run_narrative_aggregation(config: KAGConfig, *, clear_previous: bool) -> None:
    cleanup_builder = CommunityGraphBuilder(config)
    try:
        if clear_previous:
            logger.info("[Aggregation] clearing previous narrative/community aggregation nodes before narrative build")
            cleanup_builder.clear_community_aggregation()
            cleanup_builder.clear_narrative_aggregation()
    finally:
        cleanup_builder.close()

    narrative_builder = NarrativeGraphBuilder(config)
    narrative_builder.extract_episodes(
        limit_documents=50,
        document_concurrency=64,
        store_episode_support_edges=True,
        ensure_episode_embeddings=True,
        embedding_text_field="name_desc",
        embedding_batch_size=256,
    )
    narrative_builder.extract_episode_relations(
        episode_pair_concurrency=64,
        max_episode_pairs_global=200000,
        cross_document_only=False,
        similarity_threshold=float(getattr(config.narrative_graph_builder, "episode_relation_similarity_threshold", 0.55) or 0.55),
        ensure_episode_embeddings=True,
        show_pair_progress=True,
        save_pair_json=True,
        embedding_text_field="name_desc",
        embedding_batch_size=256,
    )
    narrative_builder.break_episode_cycles(method="saber")
    narrative_builder.build_storyline_candidates(
        method="trie",
        min_trunk_len=2,
    )
    narrative_builder.extract_storylines_from_candidates(
        ensure_storyline_embeddings=True,
        embedding_text_field="name_desc",
        embedding_batch_size=256,
    )
    narrative_builder.extract_storyline_relations(
        similarity_threshold=0.5,
        overlap_pair_only=False,
        show_pair_progress=False,
    )
    narrative_builder.load_json_to_graph_store(
        store_episodes=True,
        store_support_edges=True,
        store_episode_relations=True,
        store_storylines=True,
        store_storyline_support_edges=True,
        store_storyline_relations=True,
    )


def run_community_aggregation(config: KAGConfig, *, clear_previous: bool) -> None:
    community_builder = CommunityGraphBuilder(config)
    try:
        result = community_builder.run(
            clear_previous_community=clear_previous,
            clear_previous_narrative=clear_previous,
        )
        logger.info("[Aggregation] community build completed: %s", result)
    finally:
        community_builder.close()


def main():
    args = parse_args()
    if not args.json_file:
        raise ValueError("--json_file is required.")

    config = KAGConfig.from_yaml(args.config)
    run_base_graph_pipeline(config, json_file_path=args.json_file)

    aggregation_mode = str(getattr(config.global_config, "aggregation_mode", "narrative") or "narrative").strip().lower()
    clear_previous = True
    if aggregation_mode == "narrative":
        run_narrative_aggregation(config, clear_previous=clear_previous)
        return
    if aggregation_mode == "community":
        run_community_aggregation(config, clear_previous=clear_previous)
        return
    if aggregation_mode == "full":
        run_narrative_aggregation(config, clear_previous=clear_previous)
        run_community_aggregation(config, clear_previous=False)
        return
    raise ValueError(f"Unsupported global.aggregation_mode: {aggregation_mode}")


if __name__ == "__main__":
    main()
