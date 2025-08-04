"""
数据模型定义 - 剧本分析优化版 (自动读取 schema)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# === 从 schema 自动导入类型定义 ===
# from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS


# === 动态生成 EntityType Enum ===
# EntityType = Enum(
#     "EntityType", {etype["type"]: etype["type"] for etype in ENTITY_TYPES}
# )

# # === 动态生成 RelationType Enum ===
# # RELATION_TYPES = []
# # for group in RELATION_TYPE_GROUPS.values():
# #     RELATION_TYPES.extend(group)

# RelationType = Enum(
#     "RelationType", {rtype["type"]: rtype["type"] for rtype in RELATION_TYPES}
# )

# === 其他模型 ===

class Entity(BaseModel):
    id: str = Field(description="实体唯一标识")
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型")
    aliases: List[str] = Field(default_factory=list, description="别名列表")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    description: Optional[str] = Field(default=None, description="实体描述")
    scope: Optional[str] = Field(default=None, description="是否为全局实体")
    # confidence: float = Field(default=1.0, description="置信度")
    source_chunks: List[str] = Field(default_factory=list, description="来源文本块")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False

class Relation(BaseModel):
    id: str = Field(description="关系唯一标识")
    subject_id: str = Field(description="主体实体ID")
    predicate: str = Field(description="关系类型")
    object_id: str = Field(description="客体实体ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")
    # confidence: float = Field(default=1.0, description="置信度")
    source_chunks: List[str] = Field(default_factory=list, description="来源文本块")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.id == other.id
        return False

class Document(BaseModel):
    id: str = Field(description="文档唯一标识")
    content: str = Field(description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        return False

class TextChunk(BaseModel):
    id: str = Field(description="文本块唯一标识")
    content: str = Field(description="文本块内容")
    document_id: str = Field(description="所属文档ID")
    start_pos: int = Field(description="在文档中的起始位置")
    end_pos: int = Field(description="在文档中的结束位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文本块元数据")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TextChunk):
            return self.id == other.id
        return False

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="抽取的实体")
    relations: List[Relation] = Field(default_factory=list, description="抽取的关系")
    chunk_id: str = Field(description="文本块ID")
    processing_time: float = Field(description="处理时间")

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        merged_entities = list(set(self.entities + other.entities))
        merged_relations = list(set(self.relations + other.relations))
        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            chunk_id=f"{self.chunk_id}+{other.chunk_id}",
            processing_time=self.processing_time + other.processing_time,
        )

class KnowledgeGraph(BaseModel):
    entities: Dict[str, Entity] = Field(default_factory=dict, description="实体字典")
    relations: Dict[str, Relation] = Field(default_factory=dict, description="关系字典")
    documents: Dict[str, Document] = Field(default_factory=dict, description="文档字典")
    chunks: Dict[str, TextChunk] = Field(default_factory=dict, description="文本块字典")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        self.updated_at = datetime.now()

    def add_relation(self, relation: Relation) -> None:
        self.relations[relation.id] = relation
        self.updated_at = datetime.now()

    def add_document(self, document: Document) -> None:
        self.documents[document.id] = document
        self.updated_at = datetime.now()

    def add_chunk(self, chunk: TextChunk) -> None:
        self.chunks[chunk.id] = chunk
        self.updated_at = datetime.now()

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for entity in self.entities.values():
            if entity.name == name or name in entity.aliases:
                return entity
        return None

    def get_relations_by_entity(self, entity_id: str) -> List[Relation]:
        relations = []
        for relation in self.relations.values():
            if relation.subject_id == entity_id or relation.object_id == entity_id:
                relations.append(relation)
        return relations

    def stats(self) -> Dict[str, int]:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }
