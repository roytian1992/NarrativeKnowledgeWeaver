from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.memory.retrieval_strategy_memory import RetrievalStrategyMemory
from core.utils.config import KAGConfig


class FakeEmbeddingModel:
    """Deterministic tiny embedding model for local smoke tests."""

    def _vec(self, text: str, dim: int = 32) -> List[float]:
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        rnd = random.Random(int.from_bytes(h[:8], "big"))
        return [rnd.uniform(-1.0, 1.0) for _ in range(dim)]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def main() -> None:
    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")

    cfg.strategy_memory.enabled = True
    cfg.strategy_memory.read_enabled = True
    cfg.strategy_memory.abstraction_mode = "rule"
    cfg.strategy_memory.library_path = "reports/strategy_memory_smoke_library.json"
    cfg.strategy_memory.report_path = "reports/strategy_memory_smoke_report.json"
    cfg.strategy_memory.min_template_support = 1

    os.makedirs(os.path.dirname(cfg.strategy_memory.library_path), exist_ok=True)
    with open(cfg.strategy_memory.library_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "library_version": 1,
                "aggregation_mode": "narrative",
                "dataset_name": "smoke",
                "templates": [
                    {
                        "template_id": "stm_relation_evidence",
                        "pattern_name": "Relation With Evidence",
                        "pattern_description": "For relation questions, retrieve graph relations before verifying with text evidence.",
                        "query_abstract": "Ask for the relation between [CHARACTER] and [CHARACTER] with evidence.",
                        "query_pattern": {
                            "query_abstract": "Ask for the relation between [CHARACTER] and [CHARACTER] with evidence.",
                            "problem_type": "relation_or_interaction_lookup",
                            "target_category": "character",
                            "answer_shape": "short_fact",
                            "retrieval_goals": ["identify relation", "retrieve evidence"],
                        },
                        "recommended_chain": ["retrieve_entity_by_name", "get_relations_between_entities", "vdb_docs_search"],
                        "support_count": 3,
                        "success_rate": 1.0,
                    },
                    {
                        "template_id": "stm_section_localization",
                        "pattern_name": "Section Localization",
                        "pattern_description": "For appearance questions, locate sections first and then verify supporting text.",
                        "query_abstract": "Locate the sections where [CHARACTER] appears.",
                        "query_pattern": {
                            "query_abstract": "Locate the sections where [CHARACTER] appears.",
                            "problem_type": "section_localization",
                            "target_category": "character",
                            "answer_shape": "section_names",
                            "retrieval_goals": ["identify relevant sections", "return precise section titles"],
                        },
                        "recommended_chain": ["retrieve_entity_by_name", "search_sections", "search_related_content"],
                        "support_count": 2,
                        "success_rate": 0.8,
                    },
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    mem = RetrievalStrategyMemory(config=cfg, llm=None, embedding_model=FakeEmbeddingModel())
    ctx = mem.prepare_read_context(query="他们的关系是什么，给证据", doc_type="screenplay", mode="hybrid")
    report = mem.export_diagnostics(cfg.strategy_memory.report_path)

    print("routing_hint:\n", ctx.get("routing_hint", ""))
    print("matched_template_ids:", [x.get("template_id") for x in ctx.get("patterns", [])])
    print("report_template_count:", report.get("template_count"))
    print("report_path:", cfg.strategy_memory.report_path)


if __name__ == "__main__":
    main()
