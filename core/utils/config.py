# config.py
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
import yaml

# =========================
# 工具
# =========================

def _update_dc_from_dict(obj, data: Dict[str, Any], aliases: Dict[str, str] = None):
    """将 dict 中与 dataclass 字段同名（或别名映射）的键赋值给 obj。"""
    if not data:
        return
    aliases = aliases or {}
    for k, v in data.items():
        k2 = aliases.get(k, k)
        if hasattr(obj, k2):
            setattr(obj, k2, v)

def _first_present(d: Dict[str, Any], *keys: str, default=None):
    """返回 d 中第一个存在的 key 的值，用于兼容多种字段名。"""
    if not d:
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

# =========================
# 各分段配置
# =========================

@dataclass
class KnowledgeGraphBuilderConfig:
    prompt_dir: str = "./core/prompts"
    doc_type: str = "screenplay"
    max_workers: int = 32


@dataclass
class EventPlotGraphBuilderConfig:
    max_workers: int = 32
    max_depth: int = 3
    check_weakly_connected_components: bool = True
    min_connected_component_size: int = 10
    max_num_triangles: int = 2000
    max_iterations: int = 5
    min_confidence: float = 0.5
    # 新增：回退事件类型，按优先级书写，如 ["Action", "Goal"]
    event_fallback: List[str] = field(default_factory=list)


@dataclass
class ProbingConfig:
    # fixed / adjust / from_scratch
    probing_mode: str = "fixed"
    refine_background: bool = False
    # 新增：任务目标说明（例如：面向剧本理解）
    task_goal: str = ""
    max_workers: int = 32
    max_retries: int = 2
    relation_prune_threshold: float = 0.05
    entity_prune_threshold: float = 0.02
    default_graph_schema_path: Optional[str] = "./core/schema/graph_schema.json"
    default_background_path: Optional[str] = "./examples/settings/we2_settings.json"
    experience_limit: int = 100


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8096
    enable_thinking: bool = False
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model_name: str = "Qwen3-Embedding-8B"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None
    timeout: int = 60


@dataclass
class RerankConfig:
    provider: str = "cohere"
    model_name: str = "Qwen3-Reranker-8B"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60


@dataclass
class DocumentProcessingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 0
    max_workers: int = 32
    max_segments: int = 3
    max_content_size: int = 2000


@dataclass
class AgentConfig:
    max_workers: int = 32
    score_threshold: int = 7
    max_retries: int = 2
    async_timeout: int = 600
    async_max_attempts: int = 3
    async_backoff_seconds: int = 60


@dataclass
class MemoryConfig:
    enabled: bool = True
    memory_type: str = "vector"
    max_token_limit: int = 4000
    memory_path: str = "./data/memory"
    history_memory_size: int = 3
    insight_memory_size: int = 10


@dataclass
class StorageConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    vector_store_type: str = "chroma"
    vector_store_path: str = "data/vector_store"

    document_store_path: str = "data/document_store"
    knowledge_graph_path: str = "data/knowledge_graph"
    graph_schema_path: str = "data/graph_schema"
    sql_database_path: str = "data/sql"


# =========================
# 顶层配置
# =========================

@dataclass
class KAGConfig:
    # 任务段
    knowledge_graph_builder: KnowledgeGraphBuilderConfig = field(default_factory=KnowledgeGraphBuilderConfig)
    event_plot_graph_builder: EventPlotGraphBuilderConfig = field(default_factory=EventPlotGraphBuilderConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)

    # 模型段
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph_embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)      # 图谱向量
    vectordb_embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)   # 检索向量
    rerank: RerankConfig = field(default_factory=RerankConfig)

    # 流水线段
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # ---------- 构造 ----------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "KAGConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        # knowledge_graph_builder
        if "knowledge_graph_builder" in data:
            _update_dc_from_dict(cfg.knowledge_graph_builder, data["knowledge_graph_builder"])

        # event_plot_graph_builder
        if "event_plot_graph_builder" in data:
            _update_dc_from_dict(cfg.event_plot_graph_builder, data["event_plot_graph_builder"])
            # 轻量规范化：event_fallback -> List[str]
            ef = cfg.event_plot_graph_builder.event_fallback
            if ef is None:
                cfg.event_plot_graph_builder.event_fallback = []
            elif isinstance(ef, (str, int, float, bool)):
                cfg.event_plot_graph_builder.event_fallback = [str(ef)]
            elif isinstance(ef, tuple):
                cfg.event_plot_graph_builder.event_fallback = [str(x) for x in ef]
            elif isinstance(ef, list):
                cfg.event_plot_graph_builder.event_fallback = [str(x) for x in ef]
            else:
                # 不可识别类型时给出提醒并置空
                print(f"[KAGConfig] 提示：event_fallback 类型异常（{type(ef)}），已忽略。")
                cfg.event_plot_graph_builder.event_fallback = []

        # probing
        if "probing" in data:
            _update_dc_from_dict(cfg.probing, data["probing"])

        # llm
        if "llm" in data:
            _update_dc_from_dict(cfg.llm, data["llm"])

        # graph_embedding（向后兼容：若只有 embedding 则灌到 graph_embedding）
        if "graph_embedding" in data:
            _update_dc_from_dict(cfg.graph_embedding, data["graph_embedding"])
        elif "embedding" in data:
            _update_dc_from_dict(cfg.graph_embedding, data["embedding"])

        # vectordb_embedding（若没有则保留默认）
        if "vectordb_embedding" in data:
            _update_dc_from_dict(cfg.vectordb_embedding, data["vectordb_embedding"])

        # rerank
        if "rerank" in data:
            _update_dc_from_dict(cfg.rerank, data["rerank"])

        # document_processing
        if "document_processing" in data:
            _update_dc_from_dict(cfg.document_processing, data["document_processing"])

        # agent
        if "agent" in data:
            _update_dc_from_dict(cfg.agent, data["agent"])

        # memory
        if "memory" in data:
            _update_dc_from_dict(cfg.memory, data["memory"])

        # storage（容错 graph_scehma_path → graph_schema_path）
        if "storage" in data:
            _update_dc_from_dict(
                cfg.storage,
                data["storage"],
                aliases={"graph_scehma_path": "graph_schema_path"}
            )

        cfg._validate()
        return cfg

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        # 用 asdict 再手工保证字段名与 YAML 结构一致
        return {
            "knowledge_graph_builder": asdict(self.knowledge_graph_builder),
            "event_plot_graph_builder": asdict(self.event_plot_graph_builder),
            "probing": asdict(self.probing),
            "llm": asdict(self.llm),
            "graph_embedding": asdict(self.graph_embedding),
            "vectordb_embedding": asdict(self.vectordb_embedding),
            "rerank": asdict(self.rerank),
            "document_processing": asdict(self.document_processing),
            "agent": asdict(self.agent),
            "memory": asdict(self.memory),
            "storage": asdict(self.storage),
        }

    def save_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False,
                           allow_unicode=True, sort_keys=False)

    # ---------- 校验 & 轻量提示 ----------

    def _validate(self) -> None:
        # probing_mode 合法值提示
        valid_modes = {"fixed", "adjust", "from_scratch"}
        if self.probing.probing_mode not in valid_modes:
            raise ValueError(f"probing.probing_mode 必须为 {valid_modes}，当前为 {self.probing.probing_mode!r}")

        # event_fallback 类型检查（仅提示，不强制）
        ef = self.event_plot_graph_builder.event_fallback
        if not isinstance(ef, list) or not all(isinstance(x, str) for x in ef):
            print(f"[KAGConfig] 提示：event_fallback 应为字符串列表，当前为 {type(ef)}，已在加载时尝试规范化。")

        # 向量维度提示（仅提示，不强制）
        for name, emb in (("graph_embedding", self.graph_embedding),
                          ("vectordb_embedding", self.vectordb_embedding)):
            if emb.dimensions is None:
                pass

        # vector_store_type 提示
        valid_vs = {"chroma", "faiss", "milvus", "pgvector"}
        if self.storage.vector_store_type not in valid_vs:
            # 不强制抛错，尽量容忍；如需严格可改为 raise
            print(f"[KAGConfig] 提示：storage.vector_store_type={self.storage.vector_store_type!r} "
                  f"不在常见集合 {valid_vs} 内，请确认。")
