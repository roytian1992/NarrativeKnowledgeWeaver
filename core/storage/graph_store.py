"""
Graph database storage module

Neo4j-based knowledge graph storage.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import json
import logging
import re

from ..models.data import KnowledgeGraph, Entity, Relation
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)


def _quote_label(name: str) -> str:
    """
    Safely quote a label or relationship type for Cypher by backtick-escaping backticks.
    Works for labels and relationship types in modern Neo4j.
    """
    if name is None:
        return "`Unknown`"
    return f"`{str(name).replace('`', '``')}`"


class GraphStore:
    """Graph database storage backed by Neo4j."""

    def __init__(self, config: KAGConfig) -> None:
        """
        Initialize the GraphStore.

        Args:
            config: Global configuration object containing Neo4j connection info.
        """
        self.config = config
        self.driver = None
        self._connect()

    def _connect(self) -> None:
        """Connect to the Neo4j database and verify connectivity."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.storage.neo4j_uri,
                auth=(
                    self.config.storage.neo4j_username,
                    self.config.storage.neo4j_password,
                ),
            )
            # Smoke test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established successfully.")
        except Exception as e:
            logger.exception("Failed to connect to Neo4j: %s", str(e))
            self.driver = None

    def reset_knowledge_graph(self) -> None:
        """
        Reset the knowledge graph by deleting all nodes and relationships.
        """
        if not self.driver:
            logger.warning("Neo4j is not connected; skipping graph reset.")
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Knowledge graph has been reset (all nodes and relationships deleted).")

    def store_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """
        Store a full knowledge graph (entities and relations).

        Args:
            kg: KnowledgeGraph object containing entities and relations to upsert.
        """
        if not self.driver:
            logger.warning("Neo4j is not connected; skipping graph storage.")
            return

        with self.driver.session() as session:
            # Entities
            logger.info("Storing entities: count=%d", len(kg.entities.values()))
            for entity in kg.entities.values():
                self._store_entity(session, entity)

            # Relations
            logger.info("Storing relations: count=%d", len(kg.relations.values()))
            for relation in kg.relations.values():
                self._store_relation(session, relation)

    def _store_entity(self, session, entity: Entity) -> None:
        """
        Upsert a single entity as a node.

        Behavior:
        - If `entity.type` is a list, the first element is the primary label,
          the rest are added as extra labels.
        - Always adds a super label `:Entity` for consistent querying.
        - Stores a `types` property (list) for downstream compatibility.
        - `properties` are stored as a JSON string to keep structure intact.
        """
        try:
            # Normalize types to a unique ordered list
            if isinstance(entity.type, list):
                seen = set()
                types: List[str] = []
                for t in entity.type:
                    if t and t not in seen:
                        seen.add(t)
                        types.append(t)
            else:
                types = [entity.type] if entity.type else ["Concept"]

            primary = types[0]
            others = [t for t in types[1:] if t]

            # Build MERGE with primary label; set core attributes
            q_primary = _quote_label(primary)
            query_merge = f"""
            MERGE (e:{q_primary} {{id: $id}})
            SET e:Entity,
                e.name = $name,
                e.aliases = $aliases,
                e.description = $description,
                e.scope = $scope,
                e.types = $types,
                e.properties = $properties,
                e.source_chunks = $source_chunks
            """
            session.run(
                query_merge,
                {
                    "id": entity.id,
                    "name": entity.name,
                    "aliases": entity.aliases,
                    "description": entity.description,
                    "scope": getattr(entity, "scope", None) or "local",
                    "types": types,
                    "properties": json.dumps(entity.properties, ensure_ascii=False),
                    "source_chunks": entity.source_chunks,
                },
            )

            # Add extra labels if any
            if others:
                extra_labels = ":".join(_quote_label(t) for t in others)
                query_add_labels = f"""
                MATCH (e {{id: $id}})
                SET e:{extra_labels}
                """
                session.run(query_add_labels, {"id": entity.id})

        except Exception as e:
            logger.exception("[Neo4j] MERGE Entity failed: %s | entity=%s", e, entity)

    def _store_relation(self, session, relation: Relation) -> None:
        """
        Upsert a single relationship.

        Notes:
        - Relationship type is taken from `relation.predicate` and quoted safely.
        - `properties` are stored as a JSON string.
        - Assumes both subject and object nodes already exist (created by _store_entity).
        """
        try:
            rel_type = _quote_label(relation.predicate)
            query = f"""
            MATCH (s {{id: $subject_id}})
            MATCH (o {{id: $object_id}})
            MERGE (s)-[r:{rel_type} {{id: $id}}]->(o)
            SET r.predicate = $predicate,
                r.properties = $properties,
                r.source_chunks = $source_chunks
            """
            session.run(
                query,
                {
                    "id": relation.id,
                    "subject_id": relation.subject_id,
                    "object_id": relation.object_id,
                    "predicate": relation.predicate,
                    "properties": json.dumps(relation.properties, ensure_ascii=False),
                    "source_chunks": relation.source_chunks,
                },
            )
        except Exception as e:
            logger.exception("[Neo4j] MERGE Relation failed: %s | relation=%s", e, relation)

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """
        Search entities by name or aliases using a CONTAINS substring match.

        Args:
            query: Substring to search in `name` or any `aliases`.
            limit: Max number of entities to return.

        Returns:
            A list of `Entity` dataclass instances.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            cypher_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query OR any(alias IN coalesce(e.aliases, []) WHERE alias CONTAINS $query)
            RETURN e
            LIMIT $limit
            """
            result = session.run(cypher_query, {"query": query, "limit": limit})
            entities: List[Entity] = []

            for record in result:
                node = record["e"]
                # Read structured properties
                props_raw = node.get("properties", "{}")
                try:
                    props = json.loads(props_raw) if isinstance(props_raw, str) else (props_raw or {})
                except Exception:
                    props = {}

                # Prefer the `types` list; fall back to single `type` if present
                node_types = node.get("types")
                if node_types is None:
                    # legacy fallback
                    node_types = node.get("type")

                entity = Entity(
                    id=node["id"],
                    name=node.get("name"),
                    type=node_types,
                    aliases=node.get("aliases", []),
                    description=node.get("description"),
                    properties=props,
                    source_chunks=node.get("source_chunks", []),
                )
                entities.append(entity)

            return entities

    def search_relations(self, entity_name: str, limit: int = 10) -> List[Relation]:
        """
        Search relations connected to an entity by its exact `name` on either side.

        Args:
            entity_name: Exact entity name (subject or object).
            limit: Max number of relations to return.

        Returns:
            A list of `Relation` dataclass instances.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            cypher_query = """
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.name = $entity_name OR o.name = $entity_name
            RETURN r, s.id AS subject_id, o.id AS object_id
            LIMIT $limit
            """
            result = session.run(cypher_query, {"entity_name": entity_name, "limit": limit})
            relations: List[Relation] = []

            for record in result:
                r = record["r"]
                props_raw = r.get("properties", "{}")
                try:
                    props = json.loads(props_raw) if isinstance(props_raw, str) else (props_raw or {})
                except Exception:
                    props = {}

                relation = Relation(
                    id=r.get("id"),
                    subject_id=record["subject_id"],
                    object_id=record["object_id"],
                    predicate=r.get("predicate"),
                    properties=props,
                    source_chunks=r.get("source_chunks", []),
                )
                relations.append(relation)

            return relations

    def get_stats(self) -> Dict[str, Any]:
        """
        Get simple graph statistics.

        Returns:
            A dict with connection status and counts of nodes and relationships.
        """
        if not self.driver:
            return {"status": "disconnected"}

        with self.driver.session() as session:
            try:
                entity_count = session.run("MATCH (e) RETURN count(e) AS count").single()["count"]
                relation_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
                return {
                    "status": "connected",
                    "entities": entity_count,
                    "relations": relation_count,
                }
            except Exception as e:
                logger.exception("Failed to get stats: %s", str(e))
                return {"status": "error", "error": str(e)}

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")
