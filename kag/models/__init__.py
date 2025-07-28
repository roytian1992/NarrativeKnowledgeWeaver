"""
数据模型模块
"""

from .entities import (
    Entity,
    Relation,
    Document,
    TextChunk,
    KnowledgeGraph,
    # EntityType,
    # RelationType,
    ExtractionResult,
)
from .script_models import (
    SceneMetadata,
    DialogueData,
    ScriptDocument,
    ScriptContentParser,
)

__all__ = [
    "Entity",
    "Relation",
    "Document", 
    "TextChunk",
    "KnowledgeGraph",
    "EntityType",
    "RelationType",
    "ExtractionResult",
    "SceneMetadata",
    "DialogueData",
    "ScriptDocument",
    "ScriptContentParser",
]

