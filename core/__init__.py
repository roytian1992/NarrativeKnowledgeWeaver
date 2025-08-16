"""
KAG-Builder: 知识图谱构建器

基于LangChain的智能知识图谱构建器，专门优化支持剧本、小说等文学作品。
"""

__version__ = "1.0.0"
__author__ = "KAG Team"
__email__ = "team@kag.ai"

from .builder import KnowledgeGraphBuilder
from .models import Entity, Relation, Document, KnowledgeGraph
from .utils import KAGConfig

__all__ = [
    "KnowledgeGraphBuilder",
    "Entity",
    "Relation", 
    "Document",
    "KnowledgeGraph",
    "KAGConfig",
]

