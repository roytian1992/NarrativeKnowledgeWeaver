"""
Neo4j数据库操作工具类
提供可扩展的查询接口，便于后续添加新的查询功能
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Set
import json
import networkx as nx
from neo4j import Driver
from community import best_partition
from kag.models.entities import Entity, Relation


class Neo4jUtils:
    """
    Neo4j数据库操作工具类
    设计原则：
    1. 基础查询方法可复用
    2. 支持动态Cypher查询构建
    3. 便于后续添加新的查询功能
    4. 查询结果标准化处理
    """
    
    def __init__(self, driver: Driver):
        """
        初始化Neo4j工具类
        
        Args:
            driver: Neo4j连接驱动
        """
        self.driver = driver
    
    def execute_query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict]:
        """
        执行自定义Cypher查询的通用方法
        
        Args:
            cypher: Cypher查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        if params is None:
            params = {}
            
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]
    
    def search_entities_by_type(
        self,
        entity_type: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
    ) -> List[Entity]:
        """
        搜索图中所有满足类型和关键词的实体（可选过滤）
        
        Args:
            entity_type: 实体类型（如 "Character", "Concept", "Object"，传 None 表示不限制）
            keyword: 可选名称关键词（模糊匹配 name 或 aliases）
            limit: 返回结果上限
            
        Returns:
            List[Entity]
        """
        if self.driver is None:
            return []

        cypher_template = f"""
        MATCH (e:{entity_type if entity_type else ''})
        {{where_clause}}
        RETURN DISTINCT e
        LIMIT $limit
        """

        # 动态拼接 WHERE 子句
        where_clauses = []
        params = {"limit": limit}

        if entity_type:
            where_clauses.append("e.type = $etype")
            params["etype"] = entity_type

        if keyword:
            where_clauses.append(
                "(e.name CONTAINS $kw OR any(alias IN e.aliases WHERE alias CONTAINS $kw))"
            )
            params["kw"] = keyword

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        cypher = cypher_template.format(where_clause=where_clause)

        # 执行查询
        with self.driver.session() as session:
            result = session.run(cypher, params)
            entities = []
            for record in result:
                data = record["e"]
                entities.append(self._build_entity_from_data(data))
            return entities

    def search_related_entities(
        self,
        source_id: str,
        predicate: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
        return_relations: bool = False
    ) -> Union[List[Entity], List[Tuple[Entity, Relation]]]:
        """
        搜索与指定实体相关的实体
        
        Args:
            source_id: 源实体ID
            predicate: 关系谓词过滤
            entity_types: 目标实体类型过滤
            limit: 结果数量限制
            return_relations: 是否返回关系信息
            
        Returns:
            实体列表或实体-关系元组列表
        """
        if self.driver is None:
            return []

        params = {"source_id": source_id, "limit": limit}
        if predicate:
            params["predicate"] = predicate
        if entity_types:
            params["etypes"] = entity_types

        # entity type 过滤语句
        type_filter = "AND target.type IN $etypes" if entity_types else ""
        pred_filter = "AND rel.predicate = $predicate" if predicate else ""

        results = []

        with self.driver.session() as session:
            # 正向关系
            forward_cypher = f"""
            MATCH (source)-[rel]->(target)
            WHERE source.id = $source_id
              AND rel.predicate IS NOT NULL
              {pred_filter}
              {type_filter}
            RETURN target, rel
            LIMIT $limit
            """

            for record in session.run(forward_cypher, params):
                entity, relation = self._process_entity_relation_record(record, source_id, "forward")
                if return_relations:
                    results.append((entity, relation))
                else:
                    results.append(entity)

            # 反向关系
            backward_cypher = f"""
            MATCH (target)-[rel]->(source)
            WHERE source.id = $source_id
              AND rel.predicate IS NOT NULL
              {pred_filter}
              {type_filter}
            RETURN target, rel
            LIMIT $limit
            """

            for record in session.run(backward_cypher, params):
                entity, relation = self._process_entity_relation_record(record, source_id, "backward")
                if return_relations:
                    results.append((entity, relation))
                else:
                    results.append(entity)

        return results

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        根据 ID 精准查找一个实体节点（兼容所有标签）
        
        Args:
            entity_id: 实体的唯一 ID（例如 "entity_123456"）
            
        Returns:
            匹配的 Entity 对象，如果未找到则返回 None
        """
        cypher = """
        MATCH (e)
        WHERE e.id = $entity_id
        RETURN e
        LIMIT 1
        """
        params = {"entity_id": entity_id}

        with self.driver.session() as session:
            result = session.run(cypher, params)
            record = result.single()
            if not record:
                return None

            data = record["e"]
            return self._build_entity_from_data(data)

    def list_relationship_types(self) -> List[str]:
        """
        获取 Neo4j 图数据库中已存在的所有关系类型
        
        Returns:
            关系类型名称列表（去重、按字母排序）
        """
        cypher = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        ORDER BY relationshipType
        """

        with self.driver.session() as session:
            result = session.run(cypher)
            rel_types = [record["relationshipType"] for record in result]

        return rel_types

    def has_path_between(
        self, 
        src_id: str, 
        dst_id: str, 
        max_depth: int = 5, 
        allowed_rels: Optional[List[str]] = None
    ) -> bool:
        """
        判断图中是否存在从 src 到 dst 的路径，仅允许使用白名单中指定的边类型
        
        Args:
            src_id: 源实体ID
            dst_id: 目标实体ID
            max_depth: 最大路径深度
            allowed_rels: 允许的关系类型（如 ['follows', 'supports']）
            
        Returns:
            是否存在路径
        """
        if not allowed_rels:
            print("⚠️ 没有指定 allowed_rels 白名单，查询可能无意义")
            return False

        # 用冒号拼接：:rel1|rel2|rel3
        rel_pattern = ":" + "|".join(allowed_rels)

        cypher = f"""
        MATCH p = (src {{id: $src}})-[{rel_pattern}*1..{max_depth}]-(dst {{id: $dst}})
        WHERE src.id <> dst.id
        RETURN count(p) > 0 AS connected
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    {"src": src_id, "dst": dst_id}
                ).single()
                return result["connected"] if result else False
        except Exception as e:
            print(f"[Neo4j] has_path_between (whitelist mode) 执行失败: {e}")
            return False

    def build_filtered_graph(self, allowed_rels: Set[str]) -> nx.Graph:
        """
        从 Neo4j 构建一个仅包含指定关系类型的无向图
        
        Args:
            allowed_rels: 允许的关系类型集合
            
        Returns:
            NetworkX无向图
        """
        G = nx.Graph()
        with self.driver.session() as session:
            cypher = f"""
            MATCH (s)-[r]->(o)
            WHERE type(r) IN $allowed_rels
            RETURN s.id AS src, o.id AS dst
            """
            result = session.run(cypher, {"allowed_rels": list(allowed_rels)})
            for record in result:
                src, dst = record["src"], record["dst"]
                if src and dst:
                    G.add_edge(src, dst)
        return G

    def assign_components_and_communities(self, G: nx.Graph) -> Dict[str, Tuple[int, int]]:
        """
        为图中的每个节点分配 (component_id, community_id)
        
        Args:
            G: NetworkX图
            
        Returns:
            节点ID到(连通体ID, 社区ID)的映射
        """
        node_map = {}
        component_id = 0

        for component_nodes in nx.connected_components(G):
            subgraph = G.subgraph(component_nodes)
            community_dict = best_partition(subgraph)
            for node_id in community_dict:
                community_id = community_dict[node_id]
                node_map[node_id] = (component_id, community_id)
            component_id += 1

        return node_map

    def has_path_between_nx(
        self, 
        G: nx.Graph, 
        src_id: str, 
        dst_id: str, 
        max_depth: int = 3
    ) -> bool:
        """
        判断 NetworkX 图中两个节点之间是否存在路径，且路径长度不超过 max_depth
        
        Args:
            G: NetworkX图（已过滤后的白名单图）
            src_id: 起点节点 ID
            dst_id: 终点节点 ID
            max_depth: 路径最大深度
            
        Returns:
            是否存在满足条件的路径
        """
        if src_id not in G or dst_id not in G:
            return False
        try:
            length = nx.shortest_path_length(G, source=src_id, target=dst_id)
            return length <= max_depth
        except nx.NetworkXNoPath:
            return False
        except nx.NodeNotFound:
            return False

    def _build_entity_from_data(self, data) -> Entity:
        """
        从Neo4j查询结果构建Entity对象
        
        Args:
            data: Neo4j节点数据
            
        Returns:
            Entity对象
        """
        return Entity(
            id=data["id"],
            name=data["name"],
            type=data.get("type", "Unknown"),
            aliases=data.get("aliases", []),
            description=data.get("description", ""),
            properties=json.loads(data.get("properties", "{}")),
            source_chunks=data.get("source_chunks", []),
        )

    def _process_entity_relation_record(
        self, 
        record, 
        source_id: str, 
        direction: str
    ) -> Tuple[Entity, Relation]:
        """
        处理实体-关系查询记录
        
        Args:
            record: Neo4j查询记录
            source_id: 源实体ID
            direction: 关系方向 ("forward" 或 "backward")
            
        Returns:
            (Entity, Relation)元组
        """
        data = record["target"]
        rel = record["rel"]
        
        entity = self._build_entity_from_data(data)
        
        if direction == "forward":
            relation = Relation(
                id=rel.get("id"),
                subject_id=source_id,
                predicate=rel.get("predicate"),
                object_id=data["id"],
                source_chunks=rel.get("source_chunks", []),
                properties=json.loads(rel.get("properties", "{}")),
            )
        else:  # backward
            relation = Relation(
                id=rel.get("id"),
                subject_id=data["id"],
                predicate=rel.get("predicate"),
                object_id=source_id,
                source_chunks=rel.get("source_chunks", []),
                properties=json.loads(rel.get("properties", "{}")),
            )
        
        return entity, relation

