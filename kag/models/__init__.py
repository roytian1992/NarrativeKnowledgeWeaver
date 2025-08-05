"""
数据模型模块
"""

from .data import (
    Entity,
    Relation,
    Document,
    TextChunk,
    KnowledgeGraph,
    # EntityType,
    # RelationType,
    ExtractionResult,
)

__all__ = [
    "Entity",
    "Relation",
    "Document", 
    "TextChunk",
    "KnowledgeGraph",
    "EntityType",
    "RelationType",
    "ExtractionResult"
]

