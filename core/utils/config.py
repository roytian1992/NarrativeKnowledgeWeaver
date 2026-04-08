from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, Optional, List
import json
import yaml


def _update_dc_from_dict(obj: Any, data: Dict[str, Any], aliases: Dict[str, str] | None = None) -> None:
    """
    Assign keys from dict to dataclass fields (supporting aliases).
    Key fix:
      - If target attribute is a dataclass and incoming value is a dict,
        recursively update instead of overwriting the dataclass with a dict.
    """
    if not data:
        return
    aliases = aliases or {}
    for k, v in data.items():
        k2 = aliases.get(k, k)
        if not hasattr(obj, k2):
            continue

        cur = getattr(obj, k2)

        # IMPORTANT: do not overwrite nested dataclass with dict
        if is_dataclass(cur) and isinstance(v, dict):
            _update_dc_from_dict(cur, v)
            continue

        setattr(obj, k2, v)


def _as_list_str(x: Any, default: Optional[List[str]] = None) -> List[str]:
    if default is None:
        default = []
    if x is None:
        return list(default)
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, tuple):
        return [str(i) for i in x]
    if isinstance(x, (str, int, float, bool)):
        return [str(x)]
    return list(default)


def _parse_mapping_maybe(s: Any) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None

    raw = s.strip()
    if not raw:
        return None

    try:
        obj = yaml.safe_load(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    repaired = raw
    repaired = repaired.replace(",\n}", "\n}").replace(",}", "}")
    repaired = repaired.replace(",\n]", "\n]").replace(",]", "]")
    try:
        obj = yaml.safe_load(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not d:
        return default
    return d.get(key, default)


def _resolve_language_paths(language: str, task_specs_root: str) -> Dict[str, str]:
    loc = str(language or "").strip().lower()
    root = str(task_specs_root or "").strip() or "./task_specs"
    root = root.rstrip("/")
    if loc == "en":
        return {
            "prompt_dir": f"{root}/prompts_en",
            "task_dir": f"{root}/task_settings_en",
            "schema_dir": f"{root}/schema_en",
            "text_resource_dir": f"{root}/text_resources_en",
            "tool_metadata_dir": f"{root}/tool_metadata/en",
        }
    if loc == "zh":
        return {
            "prompt_dir": f"{root}/prompts",
            "task_dir": f"{root}/task_settings",
            "schema_dir": f"{root}/schema",
            "text_resource_dir": f"{root}/text_resources",
            "tool_metadata_dir": f"{root}/tool_metadata/zh",
        }
    raise ValueError(f"Unsupported global.language='{language}'. Expected one of: en, zh")


def _apply_global_locale_paths(global_cfg: "GlobalConfig") -> None:
    lang = str(getattr(global_cfg, "language", "") or "").strip().lower()
    if not lang:
        lang = str(getattr(global_cfg, "locale", "") or "").strip().lower()
    if not lang:
        lang = "en"
    # keep backward-compat mirror field
    global_cfg.language = lang
    global_cfg.locale = lang
    paths = _resolve_language_paths(lang, global_cfg.task_specs_root)
    global_cfg.prompt_dir = paths["prompt_dir"]
    global_cfg.task_dir = paths["task_dir"]
    global_cfg.schema_dir = paths["schema_dir"]
    global_cfg.text_resource_dir = paths["text_resource_dir"]
    global_cfg.tool_metadata_dir = paths["tool_metadata_dir"]


# =========================
# Section Configs
# =========================
@dataclass
class GlobalConfig:
    language: str = "en"  # en | zh
    # deprecated alias; kept for backward compatibility with old configs/code
    locale: str = ""
    task_specs_root: str = "./task_specs"
    doc_type: str = "general"
    aggregation_mode: str = "narrative"  # narrative | community | full
    retriever_agent_backend: str = "langchain"  # langchain | qwen | openai_agents

    # Derived paths (resolved from language + task_specs_root in from_yaml)
    prompt_dir: str = "./task_specs/prompts_en"
    task_dir: str = "./task_specs/task_settings_en"
    schema_dir: str = "./task_specs/schema_en"
    text_resource_dir: str = "./task_specs/text_resources_en"
    tool_metadata_dir: str = "./task_specs/tool_metadata/en"


@dataclass
class DocumentProcessingConfig:
    chunk_size: int = 600
    chunk_overlap: int = 0
    max_workers: int = 64
    max_segments: int = 3
    max_content_size: int = 2000
    store_vector_chunks: bool = True
    sentence_chunk_size: int = 200
    sentence_chunk_overlap: int = 50
    bm25_chunk_size: int = 250
    reset_vector_collections: bool = True


@dataclass
class KnowledgeGraphBuilderConfig:
    file_path: str = "data/knowledge_graph"
    max_workers: int = 64
    max_retries: int = 2
    per_task_timeout: int = 2400


@dataclass
class StorageConfig:
    graph_store_path: str = ""

    vector_store_type: str = "chroma"
    vector_store_path: str = "data/vector_store"

    sql_database_path: str = "data/sql"


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_name: str = "Qwen3-235B"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model_name: str = "bge-m3"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 8192
    dimensions: Optional[int] = 1024
    timeout: int = 60


@dataclass
class RerankConfig:
    provider: str = "cohere"
    model_name: str = "bge-m3-reranker"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120


# ---------- Narrative graph builder nested configs ----------
@dataclass
class CausalGraphConfig:
    tau_conf: float = 0.0
    tau_eff: float = 0.0
    unified_pred: str = "EPISODE_CAUSAL_LINK"
    skip_types: List[str] = field(default_factory=lambda: ["contrasts_with", "conflicts_with"])
    flipped_types: List[str] = field(default_factory=lambda: ["elaborates"])
    # In YAML you used a block string. We normalize it to Dict[str, float] when possible.
    type_weight: Dict[str, float] = field(default_factory=lambda: {
        "causes": 1.0,
        "precedes": 0.7,
        "elaborates": 0.35,
    })


@dataclass
class CycleBreakConfig:
    delta_tie: float = 0.02
    max_iters: int = 10


@dataclass
class ChainExtractionConfig:
    method: str = "trie"  # trie | mpc
    min_effective_weight: float = 0.0
    min_similarity: float = 0.0
    min_common_neighbors: float = 0.0

    # bounded path extraction
    max_paths_per_source: int = 200
    max_total_paths: int = 5000
    max_depth: Optional[int] = None

    # trie trunk extraction
    trie_min_len: int = 3
    trie_include_cutpoint: bool = True
    trie_drop_contained: bool = True
    trie_keep_terminal_pairs: bool = False


@dataclass
class NarrativeGraphBuilderConfig:
    max_workers: int = 32
    file_path: str = "data/narrative_graph"
    enable_storyline_relations: bool = True
    episode_relation_similarity_threshold: float = 0.55
    episode_relation_dynamic_threshold: bool = True
    episode_relation_min_candidate_pairs: int = 10
    episode_relation_threshold_floor: float = 0.20
    episode_relation_threshold_step: float = 0.05
    episode_relation_backfill_by_similarity: bool = True
    episode_relation_backfill_max_episodes: int = 500
    causal_graph: CausalGraphConfig = field(default_factory=CausalGraphConfig)
    cycle_break: CycleBreakConfig = field(default_factory=CycleBreakConfig)
    chain_extraction: ChainExtractionConfig = field(default_factory=ChainExtractionConfig)


@dataclass
class CommunityGraphBuilderConfig:
    max_workers: int = 16
    file_path: str = "data/community_graph"
    projection_graph_name: str = "community_projection"
    exclude_node_labels: List[str] = field(
        default_factory=lambda: ["Scene", "Chapter", "Document", "Episode", "Storyline", "Community"]
    )
    exclude_relation_types: List[str] = field(
        default_factory=lambda: [
            "SCENE_CONTAINS",
            "CHAPTER_CONTAINS",
            "CONTAINS",
            "EPISODE_CONTAINS",
            "STORYLINE_CONTAINS",
            "COMMUNITY_CONTAINS",
            "COMMUNITY_PARENT_OF",
        ]
    )
    include_intermediate_communities: bool = True
    write_property: str = "community_id"
    intermediate_property: str = "community_path"
    relationship_weight_property: str = ""
    use_confidence_as_weight: bool = False
    min_community_size: int = 3
    gamma: float = 1.0
    theta: float = 0.01
    tolerance: float = 0.0001
    max_levels: int = 10
    concurrency: int = 4
    random_seed: int = 42
    summary_max_length: int = 300
    summary_max_members: int = 40
    summary_max_relations: int = 80
    store_summary_embeddings: bool = True
    report_max_length: int = 1800
    report_max_findings: int = 6
    report_max_input_chars: int = 12000
    report_max_child_reports: int = 8
    report_max_child_findings: int = 3


# ---------- Extraction Memory ----------
@dataclass
class ExtractionMemoryConfig:
    enabled: bool = True
    raw_store_path: str = "data/memory/raw_memory"
    distilled_store_path: str = "data/memory/distilled_memory"
    realtime_store_path: str = "data/memory/realtime_memory"
    max_context_entries: int = 12
    max_chars_per_entry: int = 120
    similarity_threshold: float = 0.92
    flush_every_n_entries: int = 20
    flush_every_n_docs: int = 10
    min_kw_hits: int = 3
    distill_enabled: bool = True
    distill_every_n_docs: int = 20
    distill_min_new_entries: int = 100
    distill_max_source_entries: int = 200
    distill_max_new_memories: int = 30
    max_raw_context_entries: int = 5


@dataclass
class StrategyMemoryConfig:
    enabled: bool = False
    read_enabled: bool = True
    require_tool_use: bool = False
    runtime_routing_note_enabled: bool = False
    runtime_router_mode: str = "branching"  # branching | stable | qwen_like | legacy_full
    runtime_router_initial_tool_limit: int = 6
    runtime_router_escalation_tool_limit: int = 10
    hidden_tool_names: List[str] = field(default_factory=list)
    library_path: str = "data/memory/strategy/strategy_library.json"
    source_question_hint_path: str = "data/memory/strategy/source_question_hints.json"
    tool_metadata_runtime_dir: str = "data/memory/strategy/tool_metadata"
    top_k_templates: int = 5
    max_hint_lines: int = 6
    max_active_patterns: int = 2

    min_template_support: int = 1
    min_similarity: float = 0.0
    match_min_score: float = 0.5
    selection_min_score: float = 0.4
    single_agent_min_selection_score: Optional[float] = None  # legacy override
    subagent_min_selection_score: Optional[float] = None  # legacy override
    subagent_max_branches: int = 5
    match_min_margin: float = 0.06  # legacy compatibility, ignored by runtime selection
    preferred_tool_boost: float = 1.1
    merge_candidate_top_k: int = 3
    merge_min_candidate_score: float = 0.28
    consolidation_rounds: int = 1
    cluster_distill_max_members: int = 12
    question_timeout_sec: int = 900
    training_max_workers: int = 16
    question_timeout_retry_rounds: int = 2
    question_timeout_retry_workers: int = 4
    timeout_retry_max_retry_per_attempt: int = 1

    abstraction_mode: str = "hybrid"  # rule | llm | hybrid
    pattern_extractor_prompt_id: str = "memory/extract_strategy_query_pattern"
    tool_description_prompt_id: str = "memory/optimize_tool_description"
    failed_answer_reflection_prompt_id: str = "memory/reflect_failed_answer"
    retry_instruction_prompt_id: str = "memory/build_retry_instruction"
    max_retry_per_attempt: int = 3
    abstract_cache_size: int = 2048

    online_enabled: bool = False
    online_buffer_dir: str = "data/memory/online"
    online_real_trace_path: str = "data/memory/online/real_traces.jsonl"
    online_strategy_buffer_path: str = "data/memory/online/online_strategy_buffer.jsonl"
    online_failure_reflection_path: str = "data/memory/online/failure_reflections.jsonl"
    online_synthetic_qa_path: str = "data/memory/online/synthetic_qa_buffer.jsonl"
    online_judge_prompt_id: str = "memory/judge_online_answer"
    sampling_branch_planner_prompt_id: str = "memory/plan_trajectory_direction"
    online_strategy_min_score: float = 0.85
    online_max_tool_items: int = 8
    online_max_evidence_chars: int = 4000
    min_sampling_branches: int = 5
    online_runtime_mode: bool = False
    online_async_enabled: bool = True
    online_async_max_workers: int = 1

    self_bootstrap_enabled: bool = False
    self_bootstrap_prompt_id: str = "memory/generate_self_bootstrap_qa"
    self_bootstrap_max_questions: int = 3
    self_bootstrap_min_source_score: float = 0.8
    self_bootstrap_min_accept_score: float = 0.9
    self_bootstrap_sampling_attempts: int = 3

    report_path: str = ""


@dataclass
class TreeSearchConfig:
    enabled: bool = False
    simulations: int = 24
    exploration_weight: float = 1.25
    branching_factor: int = 3
    max_rollout_depth: int = 4
    min_candidates_to_trigger: int = 6
    value_llm_weight: float = 0.35
    value_semantic_weight: float = 0.40
    value_prior_weight: float = 0.25
    document_limit_per_leaf: int = 3



# =========================
# Top-level config
# =========================
@dataclass
class KAGConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    knowledge_graph_builder: KnowledgeGraphBuilderConfig = field(default_factory=KnowledgeGraphBuilderConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever_llm: LLMConfig = field(default_factory=LLMConfig)
    router_llm: LLMConfig = field(default_factory=LLMConfig)
    memory_llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)

    narrative_graph_builder: NarrativeGraphBuilderConfig = field(default_factory=NarrativeGraphBuilderConfig)
    community_graph_builder: CommunityGraphBuilderConfig = field(default_factory=CommunityGraphBuilderConfig)
    extraction_memory: ExtractionMemoryConfig = field(default_factory=ExtractionMemoryConfig)
    strategy_memory: StrategyMemoryConfig = field(default_factory=StrategyMemoryConfig)
    tree_search: TreeSearchConfig = field(default_factory=TreeSearchConfig)

    # ---------- Load ----------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "KAGConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        # global
        if "global" in data:
            g = data["global"] or {}
            _update_dc_from_dict(cfg.global_, g, aliases={"language": "language", "locale": "locale"})
            _update_dc_from_dict(cfg.global_config, g, aliases={"language": "language", "locale": "locale"})
            # backward compatibility: if only locale is configured, map it to language
            has_lang = isinstance(g, dict) and bool(str(g.get("language", "")).strip())
            has_locale = isinstance(g, dict) and bool(str(g.get("locale", "")).strip())
            if has_locale and not has_lang:
                lang_from_locale = str(g.get("locale", "")).strip()
                cfg.global_.language = lang_from_locale
                cfg.global_config.language = lang_from_locale

        if "aggregation" in data and isinstance(data["aggregation"], dict):
            legacy_mode = str((data["aggregation"] or {}).get("mode", "") or "").strip()
            global_has_agg_mode = isinstance(data.get("global"), dict) and bool(
                str((data.get("global") or {}).get("aggregation_mode", "") or "").strip()
            )
            if legacy_mode and not global_has_agg_mode:
                cfg.global_.aggregation_mode = legacy_mode
                cfg.global_config.aggregation_mode = legacy_mode
        _apply_global_locale_paths(cfg.global_)
        _apply_global_locale_paths(cfg.global_config)

        # document_processing
        if "document_processing" in data:
            _update_dc_from_dict(cfg.document_processing, data["document_processing"])

        # knowledge_graph_builder
        if "knowledge_graph_builder" in data:
            _update_dc_from_dict(cfg.knowledge_graph_builder, data["knowledge_graph_builder"])

        # storage
        if "storage" in data:
            _update_dc_from_dict(cfg.storage, data["storage"])

        # llm
        if "llm" in data:
            _update_dc_from_dict(cfg.llm, data["llm"])
        _update_dc_from_dict(cfg.retriever_llm, asdict(cfg.llm))
        _update_dc_from_dict(cfg.router_llm, asdict(cfg.llm))
        _update_dc_from_dict(cfg.memory_llm, asdict(cfg.llm))
        if "retriever_llm" in data:
            _update_dc_from_dict(cfg.retriever_llm, data["retriever_llm"])
        if "router_llm" in data:
            _update_dc_from_dict(cfg.router_llm, data["router_llm"])
        if "memory_llm" in data:
            _update_dc_from_dict(cfg.memory_llm, data["memory_llm"])

        # embedding
        # allow backward compat keys: embedding / graph_embedding / vectordb_embedding
        if "embedding" in data:
            _update_dc_from_dict(cfg.embedding, data["embedding"])
        elif "graph_embedding" in data:
            _update_dc_from_dict(cfg.embedding, data["graph_embedding"])

        # rerank
        if "rerank" in data:
            _update_dc_from_dict(cfg.rerank, data["rerank"])

        # narrative_graph_builder
        if "narrative_graph_builder" in data:
            ngb = data["narrative_graph_builder"] or {}
            _update_dc_from_dict(cfg.narrative_graph_builder, ngb)

            # causal_graph
            if "causal_graph" in ngb:
                cg = ngb["causal_graph"] or {}
                _update_dc_from_dict(cfg.narrative_graph_builder.causal_graph, cg)

                # normalize list fields
                cfg.narrative_graph_builder.causal_graph.skip_types = _as_list_str(
                    cfg.narrative_graph_builder.causal_graph.skip_types,
                    default=["contrasts_with", "conflicts_with"],
                )
                cfg.narrative_graph_builder.causal_graph.flipped_types = _as_list_str(
                    cfg.narrative_graph_builder.causal_graph.flipped_types,
                    default=["elaborates"],
                )

                # normalize type_weight: accept dict or string
                tw_raw = cg.get("type_weight", None)
                if tw_raw is not None:
                    tw = _parse_mapping_maybe(tw_raw)
                    if tw is not None:
                        # coerce values to float when possible
                        out: Dict[str, float] = {}
                        for k, v in tw.items():
                            try:
                                out[str(k)] = float(v)
                            except Exception:
                                # keep best-effort default if value is bad
                                pass
                        if out:
                            cfg.narrative_graph_builder.causal_graph.type_weight = out
                    else:
                        # keep existing defaults, but give a helpful hint
                        print("[KAGConfig] Warning: narrative_graph_builder.causal_graph.type_weight cannot be parsed, using defaults.")

            # cycle_break
            if "cycle_break" in ngb:
                _update_dc_from_dict(cfg.narrative_graph_builder.cycle_break, ngb["cycle_break"] or {})

            # chain_extraction
            if "chain_extraction" in ngb:
                _update_dc_from_dict(cfg.narrative_graph_builder.chain_extraction, ngb["chain_extraction"] or {})

        # community_graph_builder
        if "community_graph_builder" in data:
            _update_dc_from_dict(cfg.community_graph_builder, data["community_graph_builder"] or {})

        # extraction_memory
        if "extraction_memory" in data:
            _update_dc_from_dict(cfg.extraction_memory, data["extraction_memory"])

        # strategy_memory
        if "strategy_memory" in data:
            _update_dc_from_dict(cfg.strategy_memory, data["strategy_memory"])
        if "tree_search" in data:
            _update_dc_from_dict(cfg.tree_search, data["tree_search"])

        cfg._validate()
        return cfg

    # ---------- Dump ----------
    def to_dict(self) -> Dict[str, Any]:
        # Match your YAML layout. Note: "global" is a reserved keyword in Python, so the field is global_.
        return {
            "global": asdict(self.global_config),
            "document_processing": asdict(self.document_processing),
            "knowledge_graph_builder": asdict(self.knowledge_graph_builder),
            "storage": asdict(self.storage),
            "llm": asdict(self.llm),
            "retriever_llm": asdict(self.retriever_llm),
            "router_llm": asdict(self.router_llm),
            "memory_llm": asdict(self.memory_llm),
            "embedding": asdict(self.embedding),
            "rerank": asdict(self.rerank),
            "narrative_graph_builder": asdict(self.narrative_graph_builder),
            "community_graph_builder": asdict(self.community_graph_builder),
            "extraction_memory": asdict(self.extraction_memory),
            "strategy_memory": asdict(self.strategy_memory),
            "tree_search": asdict(self.tree_search),
        }

    def save_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # ---------- Validate ----------
    def _validate(self) -> None:
        valid_vs = {"chroma", "faiss", "milvus", "pgvector"}
        if self.storage.vector_store_type not in valid_vs:
            print(
                f"[KAGConfig] Warning: storage.vector_store_type={self.storage.vector_store_type!r} "
                f"is not in common set {sorted(valid_vs)}. Please verify."
            )

        if self.global_config.aggregation_mode not in {"narrative", "community", "full"}:
            print("[KAGConfig] Warning: global.aggregation_mode should be one of {narrative,community,full}.")

        # lightweight sanity for type_weight
        tw = self.narrative_graph_builder.causal_graph.type_weight
        if not isinstance(tw, dict) or not tw:
            print("[KAGConfig] Warning: causal_graph.type_weight is empty or invalid, please verify.")

        sm = self.strategy_memory
        if sm.top_k_templates <= 0:
            print("[KAGConfig] Warning: strategy_memory.top_k_templates should be > 0.")
        if sm.max_active_patterns <= 0:
            print("[KAGConfig] Warning: strategy_memory.max_active_patterns should be > 0.")
        if sm.max_hint_lines < 2:
            print("[KAGConfig] Warning: strategy_memory.max_hint_lines should be >= 2.")
        if sm.match_min_margin < 0.0:
            print("[KAGConfig] Warning: strategy_memory.match_min_margin should be >= 0.")
        if sm.merge_candidate_top_k <= 0:
            print("[KAGConfig] Warning: strategy_memory.merge_candidate_top_k should be > 0.")
        if sm.cluster_distill_max_members <= 0:
            print("[KAGConfig] Warning: strategy_memory.cluster_distill_max_members should be > 0.")
        if sm.abstraction_mode not in {"rule", "llm", "hybrid"}:
            print("[KAGConfig] Warning: strategy_memory.abstraction_mode should be one of {rule,llm,hybrid}.")
        if sm.self_bootstrap_max_questions <= 0:
            print("[KAGConfig] Warning: strategy_memory.self_bootstrap_max_questions should be > 0.")

        ts = self.tree_search
        if ts.simulations <= 0:
            print("[KAGConfig] Warning: tree_search.simulations should be > 0.")
        if ts.branching_factor <= 0:
            print("[KAGConfig] Warning: tree_search.branching_factor should be > 0.")
        if ts.max_rollout_depth <= 0:
            print("[KAGConfig] Warning: tree_search.max_rollout_depth should be > 0.")

    def get_llm_profile(self, name: str = "llm") -> LLMConfig:
        key = str(name or "llm").strip().lower()
        if key in {"retriever", "retriever_llm"}:
            return self.retriever_llm
        if key in {"router", "router_llm"}:
            return self.router_llm
        if key in {"memory", "memory_llm"}:
            return self.memory_llm
        return self.llm


# Optional convenience: load a config quickly
def load_config(yaml_path: str) -> KAGConfig:
    return KAGConfig.from_yaml(yaml_path)
