"""
core.memory

Keep package import side effects minimal to avoid circular imports between
runtime memory, strategy training, and retriever-agent modules.
"""

from importlib import import_module

__all__ = [
    "BaseMemoryStore",
    "ExtractionMemoryStore",
    "MemoryDistiller",
    "OnlineStrategyBuffer",
    "QueryAbstractor",
    "RetrievalStrategyMemory",
    "SelfBootstrapManager",
    "StrategyTemplateLibrary",
]


_LAZY_IMPORTS = {
    "BaseMemoryStore": ("core.memory.base_memory", "BaseMemoryStore"),
    "ExtractionMemoryStore": ("core.memory.extraction_store", "ExtractionMemoryStore"),
    "MemoryDistiller": ("core.memory.memory_distiller", "MemoryDistiller"),
    "OnlineStrategyBuffer": ("core.memory.online_strategy_buffer", "OnlineStrategyBuffer"),
    "QueryAbstractor": ("core.memory.query_abstractor", "QueryAbstractor"),
    "RetrievalStrategyMemory": ("core.memory.retrieval_strategy_memory", "RetrievalStrategyMemory"),
    "SelfBootstrapManager": ("core.memory.self_bootstrap_manager", "SelfBootstrapManager"),
    "StrategyTemplateLibrary": ("core.memory.strategy_library", "StrategyTemplateLibrary"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
