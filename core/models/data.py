"""
Data model definitions — Screenplay Analysis (schema-aware)
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

# === Core Models ===

class Entity(BaseModel):
    id: str = Field(description="Unique identifier of the entity")
    name: str = Field(description="Entity name")
    type: Union[str, List[str]] = Field(description="Entity type (string or list of types)")
    aliases: List[str] = Field(default_factory=list, description="Alias list")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes/properties")
    description: Optional[str] = Field(default=None, description="Entity description")
    scope: Optional[str] = Field(default=None, description="Entity scope (e.g., global/local)")
    version: Optional[str] = Field(default="default", description="Screenplay Version")
    source_chunks: List[str] = Field(default_factory=list, description="Source text chunk IDs")
    additional_chunks: List[str] = Field(default_factory=list, description="Additional related text chunk IDs")
    total_degree: Optional[float] = Field(default=0.0, description="Total degree of the entity")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


class Relation(BaseModel):
    id: str = Field(description="Unique identifier of the relation")
    subject_id: str = Field(description="Subject entity ID")
    predicate: str = Field(description="Relation predicate/type")
    object_id: str = Field(description="Object entity ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relation attributes/properties")
    version: Optional[str] = Field(default="default", description="Screenplay Version")
    source_chunks: List[str] = Field(default_factory=list, description="Source text chunk IDs")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.id == other.id
        return False


class Document(BaseModel):
    id: str = Field(description="Unique identifier of the document")
    content: str = Field(description="Document content")
    version: Optional[str] = Field(default="default", description="Screenplay Version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        return False


class TextChunk(BaseModel):
    id: str = Field(description="Unique identifier of the text chunk")
    content: str = Field(description="Chunk content")
    document_id: str = Field(description="Parent document ID")
    start_pos: int = Field(description="Start position within the document")
    end_pos: int = Field(description="End position within the document")
    version: Optional[str] = Field(default="default", description="Screenplay Version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TextChunk):
            return self.id == other.id
        return False


class KnowledgeGraph(BaseModel):
    entities: Dict[str, Entity] = Field(default_factory=dict, description="Entity dictionary keyed by entity ID")
    relations: Dict[str, Relation] = Field(default_factory=dict, description="Relation dictionary keyed by relation ID")
    documents: Dict[str, Document] = Field(default_factory=dict, description="Document dictionary keyed by document ID")
    chunks: Dict[str, TextChunk] = Field(default_factory=dict, description="Text chunk dictionary keyed by chunk ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last updated time")

    def add_entity(self, entity: Entity) -> None:
        """Insert or update an entity, refreshing the 'updated_at' timestamp."""
        self.entities[entity.id] = entity
        self.updated_at = datetime.now()

    def add_relation(self, relation: Relation) -> None:
        """Insert or update a relation, refreshing the 'updated_at' timestamp."""
        self.relations[relation.id] = relation
        self.updated_at = datetime.now()

    def add_document(self, document: Document) -> None:
        """Insert or update a document, refreshing the 'updated_at' timestamp."""
        self.documents[document.id] = document
        self.updated_at = datetime.now()

    def add_chunk(self, chunk: TextChunk) -> None:
        """Insert or update a text chunk, refreshing the 'updated_at' timestamp."""
        self.chunks[chunk.id] = chunk
        self.updated_at = datetime.now()

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find an entity by exact name or alias; return None if not found."""
        for entity in self.entities.values():
            if entity.name == name or name in entity.aliases:
                return entity
        return None

    def get_relations_by_entity(self, entity_id: str) -> List[Relation]:
        """Return relations where the given entity participates as subject or object."""
        relations = []
        for relation in self.relations.values():
            if relation.subject_id == entity_id or relation.object_id == entity_id:
                relations.append(relation)
        return relations

    def stats(self) -> Dict[str, int]:
        """Basic statistics of the current graph."""
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }
