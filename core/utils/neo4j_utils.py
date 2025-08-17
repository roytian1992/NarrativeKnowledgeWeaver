"""
Neo4j数据库操作工具类
提供可扩展的查询接口，便于后续添加新的查询功能
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Set
import json
from neo4j import Driver
from core.models.data import Entity, Relation
from tqdm import tqdm
import numpy as np
from core.utils.config import EmbeddingConfig
from core.utils.format import DOC_TYPE_META

EVENT_PLOT_GRAPH_RELS = ["EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF", "HAS_EVENT"]

class Neo4jUtils:
    """
    Neo4j数据库操作工具类
    设计原则：
    1. 基础查询方法可复用
    2. 支持动态Cypher查询构建
    3. 便于后续添加新的查询功能
    4. 查询结果标准化处理
    """
    
    def __init__(self, driver: Driver, doc_type: str = "screenplay"):
        """
        初始化Neo4j工具类
        
        Args:
            driver: Neo4j连接驱动
        """
        if doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {doc_type}")
        self.doc_type = doc_type
        self.meta = DOC_TYPE_META[doc_type]
        
        self.driver = driver
        self.model = None
        self.embedding_field = "embedding"
        # self.load_emebdding_model()
        
    def load_embedding_model(self, config: EmbeddingConfig):
        if config.provider == "openai":
            from core.model_providers.openai_embedding import OpenAIEmbeddingModel
            self.model = OpenAIEmbeddingModel(config)
            self.dim = config.dimensions
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.model_name)
            self.dim = config.dimensions or self.model.get_sentence_embedding_dimension()
            
        # return model
    
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
        keyword: Optional[str] = None
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
        """

        # 动态拼接 WHERE 子句
        where_clauses = []
        params = {}

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
        relation_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
        return_relations: bool = False
    ) -> Union[List[Entity], List[Tuple[Entity, Relation]]]:
        """
        搜索与指定实体相关的实体，可按关系类型、谓词、目标实体类型过滤

        Args:
            source_id: 源实体 ID
            predicate: 关系谓词过滤（rel.predicate）
            relation_types: 关系类型标签列表（Cypher 中的 :TYPE 标签）
            entity_types: 目标实体类型过滤（target.type）
            limit: 返回数量限制（可选，不传则不限制）
            return_relations: 是否返回 (实体, 关系) 对

        Returns:
            实体列表或实体-关系元组列表
        """
        if self.driver is None:
            return []

        params: Dict[str, any] = {"source_id": source_id}
        if predicate:
            params["predicate"] = predicate
        if relation_types:
            params["rel_types"] = relation_types
        if entity_types:
            params["etypes"] = entity_types
        if limit:
            params["limit"] = limit

        # 构造 Cypher 过滤子句
        predicate_filter = "AND rel.predicate = $predicate" if predicate else ""
        # type_filter = "AND type(target) IN $etypes" if entity_types else ""
        type_filter = "AND ANY(l IN labels(target) WHERE l IN $etypes)" if entity_types else ""

        rel_type_filter = "AND type(rel) IN $rel_types" if relation_types else ""
        limit_clause = "LIMIT $limit" if limit else ""

        results = []

        with self.driver.session() as session:
            # 正向边查询
            forward_cypher = f"""
            MATCH (source)-[rel]->(target)
            WHERE source.id = $source_id
            AND rel.predicate IS NOT NULL
            {predicate_filter}
            {rel_type_filter}
            {type_filter}
            RETURN target, rel
            {limit_clause}
            """

            for record in session.run(forward_cypher, params):
                entity, relation = self._process_entity_relation_record(record, source_id, "forward")
                results.append((entity, relation) if return_relations else entity)

            # 反向边查询
            backward_cypher = f"""
            MATCH (target)-[rel]->(source)
            WHERE source.id = $source_id
            AND rel.predicate IS NOT NULL
            {predicate_filter}
            {rel_type_filter}
            {type_filter}
            RETURN target, rel
            {limit_clause}
            """

            for record in session.run(backward_cypher, params):
                entity, relation = self._process_entity_relation_record(record, source_id, "backward")
                results.append((entity, relation) if return_relations else entity)

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
        
        
    def delete_relation_by_ids(
        self,
        source_id: str,
        target_id: str,
        relation_type: str
    ) -> bool:
        """
        根据 source_id、target_id 和 relation_type 删除指定关系

        Args:
            source_id: 源实体的 ID
            target_id: 目标实体的 ID
            relation_type: 要删除的关系类型（如 "EVENT_CAUSES"）

        Returns:
            bool: 是否成功删除了关系（True 表示至少删除了一条）
        """
        cypher = f"""
        MATCH (s)-[r:{relation_type}]->(t)
        WHERE s.id = $source_id AND t.id = $target_id
        DELETE r
        RETURN COUNT(r) AS deleted_count
        """
        params = {"source_id": source_id, "target_id": target_id}

        with self.driver.session() as session:
            result = session.run(cypher, params)
            record = result.single()
            return record and record["deleted_count"] > 0


    def get_common_neighbors(
        self,
        id1: str,
        id2: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "any",  # "any" / "out" / "in"
        limit: Optional[int] = None,
    ) -> List[Entity]:
        """
        返回两个实体的共同邻居（默认忽略方向）。
        
        Args:
            id1: 第一个实体的 e.id
            id2: 第二个实体的 e.id
            rel_types: 关系类型白名单（如 ["RELATED_TO", "LOCATED_IN"]），None 表示不限
            direction: "any"（无向）、"out"（a->n & b->n）、"in"（a<-n & b<-n）
            limit: 可选的上限条数
            
        Returns:
            List[Entity]: 共同邻居的实体列表
        """
        # 动态关系类型片段
        type_pattern = ""
        if rel_types:
            # 关系类型用 | 连接，如 :TYPE1|TYPE2
            type_pattern = ":" + "|".join(rel_types)

        # 动态方向
        if direction == "out":
            rel1 = f"-[r1{type_pattern}]->"
            rel2 = f"-[r2{type_pattern}]->"
        elif direction == "in":
            rel1 = f"<-[r1{type_pattern}]-"
            rel2 = f"<-[r2{type_pattern}]-"
        else:
            rel1 = f"-[r1{type_pattern}]-"
            rel2 = f"-[r2{type_pattern}]-"

        cypher = f"""
        MATCH (a {{id: $id1}}), (b {{id: $id2}})
        MATCH (a){rel1}(n)
        MATCH (b){rel2}(n)
        WHERE n.id <> $id1 AND n.id <> $id2
        RETURN DISTINCT n
        {"LIMIT $limit" if limit else ""}
        """

        params: Dict[str, Any] = {"id1": id1, "id2": id2}
        if limit:
            params["limit"] = limit

        with self.driver.session() as session:
            result = session.run(cypher, params)
            neighbors: List[Entity] = []
            for record in result:
                node = record["n"]
                neighbors.append(self._build_entity_from_data(node))
            return neighbors


    def get_common_neighbors_with_rels(
        self,
        id1: str,
        id2: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "any",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        返回共同邻居，并附带从 a/b 指向该邻居的关系类型列表（便于调试/分析）。
        
        Returns:
            List[Dict]: 形如
                {
                "entity": Entity,
                "rels_from_a": ["RELATED_TO", ...],
                "rels_from_b": ["LOCATED_IN", ...]
                }
        """
        type_pattern = ""
        if rel_types:
            type_pattern = ":" + "|".join(rel_types)

        if direction == "out":
            rel1 = f"-[r1{type_pattern}]->"
            rel2 = f"-[r2{type_pattern}]->"
        elif direction == "in":
            rel1 = f"<-[r1{type_pattern}]-"
            rel2 = f"<-[r2{type_pattern}]-"
        else:
            rel1 = f"-[r1{type_pattern}]-"
            rel2 = f"-[r2{type_pattern}]-"

        cypher = f"""
        MATCH (a {{id: $id1}}), (b {{id: $id2}})
        MATCH (a){rel1}(n)
        MATCH (b){rel2}(n)
        WHERE n.id <> $id1 AND n.id <> $id2
        RETURN DISTINCT n, collect(DISTINCT type(r1)) AS fromA, collect(DISTINCT type(r2)) AS fromB
        {"LIMIT $limit" if limit else ""}
        """

        params: Dict[str, Any] = {"id1": id1, "id2": id2}
        if limit:
            params["limit"] = limit

        with self.driver.session() as session:
            result = session.run(cypher, params)
            out: List[Dict[str, Any]] = []
            for record in result:
                node = record["n"]
                out.append({
                    "entity": self._build_entity_from_data(node),
                    "rels_from_a": record["fromA"] or [],
                    "rels_from_b": record["fromB"] or [],
                })
            return out


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
    
    def list_entity_types(self) -> List[str]:
        """
        获取 Neo4j 图数据库中已存在的所有实体类型（节点标签）

        Returns:
            实体类型名称列表（去重、按字母排序）
        """
        cypher = """
        CALL db.labels() YIELD label
        RETURN label
        ORDER BY label
        """
        with self.driver.session() as session:
            result = session.run(cypher)
            labels = [record["label"] for record in result]
        if "*" in labels:
            labels.remove("*")
        return labels


    def get_relation_summary(self, src_id: str, tgt_id: str, relation_type: str=None) -> Optional[str]:
        """
        直接在 Neo4j 中查找 src_id 到 tgt_id 之间的特定关系，并返回格式化描述
        
        Args:
            src_id: 源实体 ID
            tgt_id: 目标实体 ID
            relation_type: 关系类型（如 "EVENT_CAUSES"）
        
        Returns:
            格式化描述字符串或 None
        """
        cypher = f"""
        MATCH (s {{id: $src_id}})-[r:{relation_type}]->(t {{id: $tgt_id}})
        RETURN r, s.id AS source_id, t.id AS target_id
        LIMIT 1
        """
        results = self.execute_query(cypher, {"src_id": src_id, "tgt_id": tgt_id})

        if not results:
            return None

        record = results[0]
        relation = record["r"]
        description = ""
        subject_name = self.get_entity_by_id(src_id).name
        subject_description = self.get_entity_by_id(src_id).description
        object_name = self.get_entity_by_id(tgt_id).name
        object_description = self.get_entity_by_id(tgt_id).description
        if relation_type in EVENT_PLOT_GRAPH_RELS:
            if relation.get("reason", ""):
                description = " 理由: " + str(relation.get("reason"))
            return f"{src_id} --> {tgt_id}\n{subject_description}-->{object_description}{description}"
            
        relation_name = relation.get("relation_name", relation.get("predicate", relation_type))
        description = ":" + relation.get("description", "无相关描述")
        return f"{subject_name}({subject_description})-{relation_name}->{object_name}({object_description}){description}"


    def delete_relation_type(self, relation_type):
        print(f"🧹 正在清除已有的 {relation_type} 关系...")
        self.execute_query(f"""
            MATCH ()-[r:{relation_type}]->()
            DELETE r
        """)
        print(f"✅ 已删除所有 {relation_type} 关系")
        

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


    def _build_entity_from_data(self, data) -> Entity:
        """
        从Neo4j查询结果构建Entity对象
        - 不再读 e.type 属性；用节点 labels 作为类型（支持多类型）
        - 其余逻辑保持不变
        """
        # 取标签（兼容 neo4j.Node 或 dict），并去掉超标签 Entity
        if hasattr(data, "labels"):
            labels = [lbl for lbl in list(data.labels) if lbl != "Entity"]
        else:
            labels = [lbl for lbl in (data.get("labels", []) or []) if lbl != "Entity"]

        # properties 仍可能是字符串化的 JSON
        raw_props = data.get("properties", "{}")
        try:
            props = json.loads(raw_props) if isinstance(raw_props, str) else (raw_props or {})
        except Exception:
            props = {}

        return Entity(
            id=data["id"],
            name=data["name"],
            type=labels if labels else "Unknown",   # ← 这里从 labels 来（Union[str, List[str]] 兼容）
            aliases=data.get("aliases", []),
            description=data.get("description", ""),
            properties=props,
            source_chunks=data.get("source_chunks", []),
        )


    # def _build_entity_from_data(self, data) -> Entity:
    #     """
    #     从Neo4j查询结果构建Entity对象
        
    #     Args:
    #         data: Neo4j节点数据
            
    #     Returns:
    #         Entity对象
    #     """
    #     return Entity(
    #         id=data["id"],
    #         name=data["name"],
    #         type=data.get("type", "Unknown"),
    #         aliases=data.get("aliases", []),
    #         description=data.get("description", ""),
    #         properties=json.loads(data.get("properties", "{}")),
    #         source_chunks=data.get("source_chunks", []),
    #     )

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
        # print("[CHECL] rel.type: ", rel.type )
        
        entity = self._build_entity_from_data(data)
        # print("[CHECK] rel: ", [k for k in rel])
        predicate = rel.get("predicate", rel.type)
        
        if direction == "forward":
            relation_id_str = f"{source_id}_{predicate}_{data["id"]}"
        else:
            relation_id_str = f"{data["id"]}_{predicate}_{source_id}"
            
        rel_id = f"rel_{hash(relation_id_str) % 1000000}"
        
        
        if direction == "forward":
            relation = Relation(
                id=rel.get("id", rel_id),
                subject_id=source_id,
                predicate=predicate,
                object_id=data["id"],
                source_chunks=rel.get("source_chunks", []),
                properties=json.loads(rel.get("properties", "{}")),
            )
        else:  # backward
            relation = Relation(
                id=rel.get("id", rel_id),
                subject_id=data["id"],
                predicate=predicate,
                object_id=source_id,
                source_chunks=rel.get("source_chunks", []),
                properties=json.loads(rel.get("properties", "{}")),
            )
        
        return entity, relation
    
    
    def encode_node_embedding(self, node: Dict) -> List[float]:
        name = node.get("name", "")
        desc = node.get("description", "")
        props = node.get("properties", "")
        try:
            props_dict = json.loads(props) if isinstance(props, str) else props
        except Exception:
            props_dict = {}

        # 构造嵌入输入
        if props_dict:
            prop_text = "；".join([f"{k}：{v}" for k, v in props_dict.items()])
            text = f"{name}{name}{name}.{desc}.{prop_text}"
        else:
            text = f"{name}{name}{name}.{desc}"
        if len(text) > 500:
            text = text[:500] # BGE最大上下文限制
        embed = self.model.encode(text)
        embed = embed.tolist() if hasattr(embed, "tolist") else embed
        return embed

    def encode_relation_embedding(self, rel: Dict) -> Optional[List[float]]:
        try:
            props = rel.get("properties", "")
            props_dict = json.loads(props) if isinstance(props, str) else props
            desc = props_dict.get("description", "")
            if desc:
                embed = self.model.encode(desc)
                embed = embed.tolist() if hasattr(embed, "tolist") else embed
                return embed
        except Exception:
            pass
        return None
    
    def fetch_all_nodes(self, node_types: List[str]) -> List[Dict]:
        results = []
        with self.driver.session() as session:
            for label in node_types:
                query = f"""
                MATCH (e:{label})
                RETURN labels(e) AS labels, e.id AS id, e.name AS name, e.description AS description, e.properties AS properties
                """
                res = session.run(query)
                results.extend([r.data() for r in res])
        return results

    def fetch_all_relations(self, relation_types: Optional[List[str]] = None) -> List[Dict]:
        """
        获取图中所有关系，支持按关系类型（predicate）过滤。

        Args:
            relation_types: 要保留的关系类型列表（如 ["happens_at", "causes"]）
                            若为 None，则返回所有关系

        Returns:
            每条边的数据字典，字段包括 predicate、id、properties
        """
        with self.driver.session() as session:
            if relation_types:
                predicate_filter = ", ".join([f"'{r}'" for r in relation_types])
                query = f"""
                MATCH ()-[r]->()
                WHERE type(r) IN [{predicate_filter}]
                RETURN type(r) AS predicate, r.id AS id, r.properties AS properties
                """
            else:
                query = """
                MATCH ()-[r]->()
                RETURN type(r) AS predicate, r.id AS id, r.properties AS properties
                """

            result = session.run(query)
            return [record.data() for record in result]

        
    def update_node_embedding(self, node_id: str, embedding: List[float]) -> None:
        with self.driver.session() as session:
            session.run(f"""
            MATCH (e) WHERE e.id = $id
            SET e.{self.embedding_field} = $embedding
            """, id=node_id, embedding=embedding)
            
    def update_relation_embedding(self, rel_id: str, embedding: List[float]) -> None:
        with self.driver.session() as session:
            session.run(f"""
            MATCH ()-[r]->() WHERE r.id = $id
            SET r.{self.embedding_field} = $embedding
            """, id=rel_id, embedding=embedding)
    
    def process_all_embeddings(self, entity_types: List[str] = [], exclude_entity_types: List[str] = []):
        """
        自动处理所有节点标签和所有边，为其生成 embedding 并写回图数据库。
        节点 embedding 输入：name + description (+ properties)
        边 embedding 输入：properties.description
        """
        # === 获取所有实体类型（标签） ===
        if not entity_types:
            entity_types = self.list_entity_types()

        # === 处理节点嵌入 ===
        print("🚀 开始处理节点嵌入...")
        for node in exclude_entity_types:
            if node in entity_types:
                entity_types.remove(node)
                
        print(f"📌 实体类型标签: {entity_types}")
        nodes = self.fetch_all_nodes(entity_types)
        for n in  tqdm(nodes, desc="Encoding Nodes", ncols=80):
            try:
                emb = self.encode_node_embedding(n)
                self.update_node_embedding(n["id"], emb)
            except Exception as e:
                print(f"⚠️ Node {n.get('id')} embedding failed:", str(e))

        print(f"✅ 节点嵌入完成，共处理 {len(nodes)} 个节点")
                
    def ensure_entity_superlabel(self):
        """
        为所有具有 embedding 的节点添加超标签 :Entity（跳过已存在标签）
        """
        query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND NOT 'Entity' IN labels(n)
        SET n:Entity
        """
        with self.driver.session() as session:
            session.run(query)
            print("[✓] 已为所有含 embedding 的节点添加超标签 :Entity")

    def create_vector_index(self, index_name="entityEmbeddingIndex", similarity="cosine"):
        """
        删除已有同名索引并重建统一向量索引
        """

        with self.driver.session() as session:
            # DROP index if exists（5.x 语法）
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            print(f"[✓] 已删除旧索引 {index_name}（如存在）")

            # 创建新索引（标准 Cypher 语法，社区版兼容）
            session.run(f"""
            CREATE VECTOR INDEX {index_name}
            FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {self.dim},
                `vector.similarity_function`: '{similarity}'
              }}
            }}
            """)
            print(f"[✓] 已创建新向量索引 {index_name} on :Entity(embedding)")

    def _query_entity_knn(self, embedding: list, top_k: int = 5):
        """
        查询与输入 embedding 向量最相似的 top-K 节点
        """
        query = """
        CALL db.index.vector.queryNodes('entityEmbeddingIndex', $top_k, $embedding)
        YIELD node, score
        RETURN node.name AS name, labels(node) AS labels, node.id AS id, score
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            result = session.run(query, {"embedding": embedding, "top_k": top_k})
            return result.data()

    def query_similar_entities(self, text: str, top_k: int = 5, normalize: bool = True):
        """
        给定自然语言 `text`，自动编码为 embedding，查询最相似的实体节点（使用 entityEmbeddingIndex）

        Args:
            text (str): 查询文本（如实体名、事件片段等）
            model: 你的 embedding 模型（需有 encode 方法）
            top_k (int): 返回前 top-k 个结果
            normalize (bool): 是否标准化向量（确保匹配 cosine 索引）

        Returns:
            List[Dict]: 包含 name、labels、id、score 的结果列表
        """
        embed = self.model.encode(text)
        embed = embed.tolist() if hasattr(embed, "tolist") else embed
         
        return self._query_entity_knn(embed, top_k=top_k)
    
    
    def compute_semantic_similarity(self, node_id_1, node_id_2):
        query = f"""
        MATCH (a {{id: '{node_id_1}'}}), (b {{id: '{node_id_2}'}})                                          
        RETURN gds.similarity.cosine(a.embedding, b.embedding) AS similarity
        """
        result = self.execute_query(query)
        return result[0].get("similarity")
    
    def compute_graph_similarity(self, node_id_1, node_id_2, field):
        query = f"""
        MATCH (a {{id: '{node_id_1}'}}), (b {{id: '{node_id_2}'}})                                          
        RETURN gds.similarity.cosine(a.{field}, b.{field}) AS graph_similarity
        """
        result = self.execute_query(query)
        return result[0].get("graph_similarity")
    
    def check_nodes_reachable(
        self,
        src_id: str,
        dst_id: str,
        max_depth: int = 3,
        excluded_rels: Optional[List[str]] = None
    ) -> bool:
        """
        判断两个任意节点之间是否存在路径，长度不超过 max_depth，且不包含某些关系类型
        
        Args:
            src_id: 起点节点 ID
            dst_id: 终点节点 ID
            max_depth: 最大允许的路径深度
            excluded_rels: 要排除的关系类型列表（如 ["SCENE_CONTAINS"]）
            
        Returns:
            是否可达（True/False）
        """
        rel_filter = ""
        if excluded_rels:
            # 构造过滤谓词：type(r) <> 'X' AND type(r) <> 'Y' ...
            rel_filter = " AND ".join([f"type(r) <> '{rel}'" for rel in excluded_rels])
            rel_filter = f"WHERE ALL(r IN relationships(p) WHERE {rel_filter})"

        query = f"""
        MATCH (n1 {{id: $src_id}}), (n2 {{id: $dst_id}})
        RETURN EXISTS {{
            MATCH p = (n1)-[*1..{max_depth}]-(n2)
            {rel_filter}
        }} AS reachable
        """
        result = self.execute_query(query, {"src_id": src_id, "dst_id": dst_id})
        if result and isinstance(result, list):
            return result[0].get("reachable", False)
        return False


    def get_entity_info(self, event_id: str, entity_type="", contain_relations=False, contain_properties=False) -> str:
        """
        获取事件的详细信息，用于因果关系检查
        Args:
            event_id: 事件ID
            
        Returns:
            格式化的事件信息字符串
        """
        event_node = self.get_entity_by_id(event_id)
        
        relation_types = self.list_relationship_types()
        
        for relation in EVENT_PLOT_GRAPH_RELS + [self.meta["contains_pred"]]:
            if relation in relation_types:
                relation_types.remove(relation)
            
        results = self.search_related_entities(
            source_id=event_id, 
            return_relations=True,
            relation_types=relation_types
        )
        
        relevant_info = []
        for result in results:
            info = self._get_relation_info(result[1])
            if info:
                relevant_info.append("- " + info)
                
        event_description = event_node.description or "无具体描述"
        if not entity_type:
            entity_type = "实体"
        
        context = f"{entity_type}名称：{event_node.name}，描述：{event_description}\n"
        if contain_relations:
            context += f"相关信息有：\n" + "\n".join(relevant_info) + "\n"
    
        if contain_properties:
            event_props = event_node.properties
            # print(event_props)
            non_empty_props = {k: v for k, v in event_props.items() if v}

            if non_empty_props:
                context += f"{entity_type}的属性如下：\n"
                for k, v in non_empty_props.items():
                    context += f"- {k}：{v}\n"

        return context
    
    
    def _get_relation_info(self, relation) -> Optional[str]:
        """
        获取关系信息的格式化字符串
        
        Args:
            relation: 关系对象
            
        Returns:
            格式化的关系信息，如果是SCENE_CONTAINS则返回None
        """
        if relation.predicate == self.meta["contains_pred"]:
            return None
            
        subject_id = relation.subject_id
        subject_name = self.get_entity_by_id(subject_id).name
        object_id = relation.object_id
        object_name = self.get_entity_by_id(object_id).name
        relation_name = relation.properties.get("relation_name", relation.predicate)
        description = relation.properties.get("description", "")
        
        return f"{subject_name}-{relation_name}->{object_name}: {description}"
    

    def create_event_causality_graph(
        self,
        graph_name: str = "event_causality_graph",
        force_refresh: bool = True,
        min_confidence: float = 0.0,
    ):
        """
        基于 Event 节点与三类关系（EVENT_CAUSES / EVENT_INDIRECT_CAUSES / EVENT_PART_OF）
        创建一个用于因果分析的 GDS 子图（有向，NATURAL 方向）。
        仅保留 coalesce(r.confidence, 0.0) >= min_confidence 的边。
        - 兼容不同 GDS 版本：不使用 relationshipProperties；提供 parameters 与内联常量两种创建方案的回退。
        """
        from neo4j.exceptions import ClientError
        rel_types = '["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]'

        def _drop_if_exists(session, name: str):
            exists = session.run("CALL gds.graph.exists($name) YIELD exists", {"name": name}).single()["exists"]
            if exists:
                session.run("CALL gds.graph.drop($name) YIELD graphName", {"name": name})
                print(f"[✓] 已删除旧图 {name}")

        def _count_edges(session, min_conf: float) -> int:
            return session.run(
                f"""
                MATCH (:Event)-[r]->(:Event)
                WHERE type(r) IN {rel_types}
                AND coalesce(r.confidence, 0.0) >= $min_conf
                RETURN count(r) AS edge_count
                """,
                {"min_conf": float(min_conf)}
            ).single()["edge_count"]

        with self.driver.session() as s:
            # 刷新
            if force_refresh:
                _drop_if_exists(s, graph_name)

            # 已存在直接返回
            exists = s.run("CALL gds.graph.exists($name) YIELD exists", {"name": graph_name}).single()["exists"]
            if exists:
                print(f"[=] 已存在图 {graph_name}，未刷新。")
                edge_count = _count_edges(s, min_confidence)
                print(f"[✓] 当前满足条件的边数量：{edge_count}")
                return

            # -------- 方案A：使用 parameters（不带 relationshipProperties）--------
            query_A = f"""
            CALL gds.graph.project.cypher(
            $name,
            'MATCH (e:Event) RETURN id(e) AS id',
            'MATCH (e1:Event)-[r]->(e2:Event)
                WHERE type(r) IN {rel_types}
                AND coalesce(r.confidence, 0.0) >= $min_conf
                AND e1 <> e2
            RETURN id(e1) AS source,
                    id(e2) AS target,
                    coalesce(r.confidence, 0.0) AS confidence',
            {{ parameters: {{ min_conf: $min_conf }} }}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """

            # -------- 方案B：不使用 parameters，内联常量 --------
            query_B = f"""
            CALL gds.graph.project.cypher(
            $name,
            'MATCH (e:Event) RETURN id(e) AS id',
            'MATCH (e1:Event)-[r]->(e2:Event)
                WHERE type(r) IN {rel_types}
                AND coalesce(r.confidence, 0.0) >= {float(min_confidence)}
                AND e1 <> e2
            RETURN id(e1) AS source,
                    id(e2) AS target,
                    coalesce(r.confidence, 0.0) AS confidence'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """

            created = False
            try:
                rec = s.run(query_A, {"name": graph_name, "min_conf": float(min_confidence)}).single()
                created = True
            except ClientError as e:
                # 某些版本不支持 parameters 键，回退到内联常量方案
                print(f"[i] 使用 parameters 方案失败，回退（原因：{str(e)[:120]} ...）")
            except Exception as e:
                print(f"[i] 使用 parameters 方案异常，回退（原因：{str(e)[:120]} ...）")

            if not created:
                rec = s.run(query_B, {"name": graph_name}).single()

            print(f"[+] 已创建因果子图 {rec['graphName']}")
            print(f"    节点数: {rec['nodeCount']}，边数: {rec['relationshipCount']}")

            # 统计
            edge_count = _count_edges(s, min_confidence)
            print(f"[✓] 当前满足条件的边数量：{edge_count}")


    
    def create_subgraph(
        self,
        graph_name: str = "subgraph_1",
        exclude_entity_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> None:
        """
        创建/刷新一个 GDS 命名子图：
        - 节点：全图节点，但会排除指定标签（默认 :Scene）
        - 边  ：排除指定关系类型（默认 SCENE_CONTAINS）
        
        Args:
            graph_name:            子图名称
            exclude_node_labels:   要排除的节点标签列表，默认 ["Scene"]
            exclude_rel_types:     要排除的关系类型列表，默认 ["SCENE_CONTAINS"]
            force_refresh:         如子图已存在，是否强制删除后重建
        """

        exclude_entity_types = exclude_entity_types or [self.meta["section_label"]]
        exclude_relation_types = exclude_relation_types or [self.meta["contains_pred"]]

        with self.driver.session() as s:
            # --- 1. 若已存在且要求刷新，则删除 ---
            exists = s.run("RETURN gds.graph.exists($name) AS ok",
                        name=graph_name).single()["ok"]
            if exists and force_refresh:
                s.run("CALL gds.graph.drop($name, false)", name=graph_name)
                exists = False
                print(f"[✓] 旧子图 {graph_name} 已删除并刷新")

            if exists:
                print(f"[✓] GDS 子图 {graph_name} 已存在，跳过创建")
                return

            # --- 2. 生成节点 / 关系 Cypher ---
            #   节点：排除指定标签
            label_filter = " AND ".join([f"NOT '{lbl}' IN labels(n)" for lbl in exclude_entity_types]) or "true"
            node_query = f"""
            MATCH (n) WHERE {label_filter}
            RETURN id(n) AS id
            """

            #   关系：排除指定类型 & 排除与被排除节点相连的边
            rel_filter = " AND ".join([f"type(r) <> '{rt}'" for rt in exclude_relation_types]) or "true"
            # 额外保证两端节点都不是被排除标签
            node_label_neg = " AND ".join([f"NOT '{lbl}' IN labels(a)" for lbl in exclude_entity_types] +
                                        [f"NOT '{lbl}' IN labels(b)" for lbl in exclude_entity_types]) or "true"

            rel_query = f"""
            MATCH (a)-[r]->(b)
            WHERE {rel_filter} AND {node_label_neg}
            RETURN id(a) AS source, id(b) AS target
            """

            # --- 3. 调用 project.cypher ---
            s.run("""
            CALL gds.graph.project.cypher(
            $name,
            $nodeQuery,
            $relQuery
            )
            """, name=graph_name, nodeQuery=node_query, relQuery=rel_query)

            print(f"[+] 已创建 GDS 子图 {graph_name}（排除标签 {exclude_entity_types}，排除边 {exclude_relation_types}）")

    def run_louvain(
        self,
        graph_name: str = "event_graph",
        write_property: str = "community",
        max_iterations: int = 20,
        force_run: bool = False
    ) -> None:
        """
        在指定子图上跑 Louvain；若已写过属性且 !force_run 则跳过
        """
        with self.driver.session() as s:
            if not force_run:
                # 快速检测是否已有社区字段
                has_prop = s.run("""
                    MATCH (n) WHERE exists(n[$prop]) RETURN n LIMIT 1
                """, prop=write_property).single()
                if has_prop:
                    print(f"[✓] 节点已存在 {write_property}，跳过 Louvain")
                    return

            s.run(f"""
            CALL gds.louvain.write($graph, {{
              writeProperty: $prop,
              maxIterations: $iters
            }});
            """, graph=graph_name, prop=write_property, iters=max_iterations)
            print(f"[+] Louvain 已完成，结果写入 `{write_property}`")

    
    # === 3. 取同社区事件对 ===
    def fetch_event_pairs_same_community(
            self,
            max_pairs: Optional[int] = None
        ) -> List[Dict[str, str]]:
        """
        返回同社区的事件对 ID 列表（不再考虑图中是否路径可达）
        """
        q = """
        MATCH (e1:Event), (e2:Event)
        WHERE e1.community = e2.community AND id(e1) < id(e2)
        RETURN e1.id AS srcId, e2.id AS dstId
        """
        if max_pairs:
            q += f"\nLIMIT {max_pairs}"
        return self.execute_query(q)


    def write_event_causes(self, rows: List[Dict[str, Any]]) -> None:
        """
        写入事件间关系（按 predicate 分三类）：
        - CAUSES          -> :EVENT_CAUSES      （relation_name=“导致”）
        - INDIRECT_CAUSES -> :EVENT_INDIRECT_CAUSES（relation_name=“间接导致”）
        - PART_OF         -> :EVENT_PART_OF     （relation_name=“属于/组成”）

        rows: [
        {"srcId": str, "dstId": str, "predicate": "CAUSES"|"INDIRECT_CAUSES"|"PART_OF",
        "reason": str, "confidence": float},
        ...
        ]
        """
        if not rows:
            return

        valid_rows = [r for r in rows if r.get("predicate") in ("CAUSES", "INDIRECT_CAUSES", "PART_OF")]
        if not valid_rows:
            print("[i] 无可写入的关系（全部为 NONE 或未知 predicate）")
            return

        cypher = """
        UNWIND $rows AS row
        MATCH (s:Event {id: row.srcId})
        MATCH (t:Event {id: row.dstId})
        WITH s, t, row

        // CAUSES
        FOREACH (_ IN CASE WHEN row.predicate = 'CAUSES' THEN [1] ELSE [] END |
        MERGE (s)-[r:EVENT_CAUSES]->(t)
        SET r.predicate     = row.predicate,
            r.reason        = row.reason,
            r.confidence    = coalesce(row.confidence, 0.0),
            r.relation_name = '导致'
        )

        // INDIRECT_CAUSES
        FOREACH (_ IN CASE WHEN row.predicate = 'INDIRECT_CAUSES' THEN [1] ELSE [] END |
        MERGE (s)-[r:EVENT_INDIRECT_CAUSES]->(t)
        SET r.predicate     = row.predicate,
            r.reason        = row.reason,
            r.confidence    = coalesce(row.confidence, 0.0),
            r.relation_name = '间接导致'
        )

        // PART_OF
        FOREACH (_ IN CASE WHEN row.predicate = 'PART_OF' THEN [1] ELSE [] END |
        MERGE (s)-[r:EVENT_PART_OF]->(t)
        SET r.predicate     = row.predicate,
            r.reason        = row.reason,
            r.confidence    = coalesce(row.confidence, 0.0),
            r.relation_name = '属于/组成'
        )
        """
        self.execute_query(cypher, {"rows": valid_rows})

        c_counts = {"CAUSES": 0, "INDIRECT_CAUSES": 0, "PART_OF": 0}
        for r in valid_rows:
            c_counts[r["predicate"]] += 1
        print(f"[+] 已写入/更新事件关系 {len(valid_rows)} 条 "
            f"(CAUSES={c_counts['CAUSES']}, INDIRECT_CAUSES={c_counts['INDIRECT_CAUSES']}, PART_OF={c_counts['PART_OF']})")

    
    def get_all_events_with_causality(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        rel_types_str = '["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]'
        cypher = f"""
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[r]->(t:Event)
        WHERE type(r) IN {rel_types_str} AND coalesce(r.confidence,0.0) >= $min_conf
        OPTIONAL MATCH (s:Event)-[r2]->(e)
        WHERE type(r2) IN {rel_types_str} AND coalesce(r2.confidence,0.0) >= $min_conf
        RETURN e.id AS event_id,
            e.name AS event_name,
            e.description AS event_description,
            coalesce(e.properties, "{{}}") AS event_properties,
            collect(DISTINCT {{target: t.id, confidence: coalesce(r.confidence,0.0), rel_type: type(r)}}) AS outgoing,
            collect(DISTINCT {{source: s.id, confidence: coalesce(r2.confidence,0.0), rel_type: type(r2)}}) AS incoming
        """
        return self.execute_query(cypher, {"min_conf": float(min_confidence)})


    def get_causality_edges_by_confidence(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        rel_types_str = '["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]'
        cypher = f"""
        MATCH (source:Event)-[r]->(target:Event)
        WHERE type(r) IN {rel_types_str} AND coalesce(r.confidence,0.0) >= $min_conf
        RETURN source.id AS source_id,
            target.id AS target_id,
            coalesce(r.confidence,0.0) AS confidence,
            type(r) AS rel_type
        """
        return self.execute_query(cypher, {"min_conf": float(min_confidence)})


    def identify_event_clusters_by_connectivity(self, min_confidence: float = 0.0) -> List[List[str]]:
        graph_name = f"event_causality_graph_{str(float(min_confidence)).replace('.', '_')}"
        try:
            self.execute_query(f"CALL gds.graph.drop('{graph_name}') YIELD graphName")
        except:
            pass

        rel_types_str = '["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]'
        create_graph_cypher = f"""
        CALL gds.graph.project.cypher(
            '{graph_name}',
            'MATCH (n:Event) RETURN id(n) AS id',
            'MATCH (a:Event)-[r]->(b:Event)
            WHERE type(r) IN {rel_types_str} AND coalesce(r.confidence,0.0) >= {float(min_confidence)}
            RETURN id(a) AS source, id(b) AS target, coalesce(r.confidence,0.0) AS confidence'
        )
        """
        self.execute_query(create_graph_cypher)

        result = self.execute_query(f"""
        CALL gds.wcc.stream('{graph_name}')
        YIELD nodeId, componentId
        RETURN gds.util.asNode(nodeId).id as event_id, componentId
        ORDER BY componentId, event_id
        """)

        clusters = {}
        for r in result:
            clusters.setdefault(r["componentId"], []).append(r["event_id"])

        # 仅保留 size>1 且节点确实出现在满足阈值的边中
        edges = self.get_causality_edges_by_confidence(min_confidence)
        connected = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
        return [c for c in clusters.values() if len(c) > 1 and any(x in connected for x in c)]

            

    def _fallback_clustering(self, threshold: float) -> List[List[str]]:
        """
        降级聚类方法：基于直接因果关系的简单聚类
        
        Args:
            threshold: 权重阈值
            
        Returns:
            List[List[str]]: 事件聚类列表
        """
        edges = self.get_causality_edges_by_confidence(threshold)
        
        # 构建邻接表
        graph = {}
        all_events = set()
        
        for edge in edges:
            source = edge['source_id']
            target = edge['target_id']
            
            all_events.add(source)
            all_events.add(target)
            
            if source not in graph:
                graph[source] = []
            if target not in graph:
                graph[target] = []
                
            graph[source].append(target)
            graph[target].append(source)  # 无向图
        
        # DFS查找连通分量
        visited = set()
        clusters = []
        
        def dfs(node, current_cluster):
            if node in visited:
                return
            visited.add(node)
            current_cluster.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, current_cluster)
        
        for event in all_events:
            if event not in visited:
                cluster = []
                dfs(event, cluster)
                if len(cluster) > 1:  # 只保留有多个事件的聚类
                    clusters.append(cluster)
        
        return clusters
    
    def enrich_event_nodes_with_context(self) -> None:
        """
        为每个 Event 节点补全上下文字段，并合并写入到 e.properties（字符串型 JSON）中：
        - time: List[str]
        - participants: List[str]
        - location: List[str]
        - chapter_name 或 scene_name: List[str]，取决于 doc_type
        """

        section_key = "scene_name" if self.doc_type == "screenplay" else "chapter_name"
        section_label = "Scene" if self.doc_type == "screenplay" else "Chapter"

        # Step 1: 查询所有事件节点及其上下文
        cypher = f"""
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[]-(t:TimePoint)
        OPTIONAL MATCH (e)-[]-(c:Character)
        OPTIONAL MATCH (e)-[]-(l:Location)
        OPTIONAL MATCH (e)-[]-(s:{section_label})
        RETURN e.id AS id,
            [x IN COLLECT(DISTINCT t.value) WHERE x IS NOT NULL] AS time,
            [x IN COLLECT(DISTINCT c.name) WHERE x IS NOT NULL] AS participants,
            [x IN COLLECT(DISTINCT l.name) WHERE x IS NOT NULL] AS location,
            [x IN COLLECT(DISTINCT s.name) WHERE x IS NOT NULL] AS {section_key},
            e.properties AS properties
        """
        records = self.execute_query(cypher)

        # Step 2: 合并字段并写入 properties（注意 properties 是字符串型 JSON）
        for r in tqdm(records, desc="更新 Event properties 上下文"):
            try:
                props: Dict[str, Any] = json.loads(r["properties"]) if r.get("properties") else {}
            except Exception:
                print(f"⚠️ JSON 解析失败，跳过 id={r['id']}")
                continue

            props["time"] = r.get("time", [])
            props["participants"] = r.get("participants", [])
            props["location"] = r.get("location", [])
            props[section_key] = r.get(section_key, [])

            self.execute_query(
                "MATCH (e:Event {id: $id}) SET e.properties = $props_str",
                {"id": r["id"], "props_str": json.dumps(props, ensure_ascii=False)}
            )

        print(f"[✓] 已将上下文属性封装写入 e.properties 字符串字段（包含 time, participants, location, {section_key}）")


    def get_event_details(self, event_ids: List[str]) -> List[Dict[str, Any]]:
        """
        返回事件节点的核心信息 + properties + 所属章节信息
        """
        cypher = f"""
        MATCH (e:Event)
        WHERE e.id IN $event_ids
        OPTIONAL MATCH (s:{self.meta['section_label']})-[:{self.meta['contains_pred']}]->(e)
        RETURN e.id          AS event_id,
            e.name        AS event_name,
            e.source_chunks AS source_chunks,
            e.description AS event_description,
            e.properties  AS event_properties,          // ← 直接返回整个属性 Map
            collect(DISTINCT s.id)   AS section_ids,
            collect(DISTINCT s.name) AS section_names
        """
        return self.execute_query(cypher, {"event_ids": event_ids})


    def get_causality_paths(self, event_ids: List[str]) -> List[Dict[str, Any]]:
        rel_types_str = '["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]'
        cypher = f"""
        MATCH (source:Event)-[r]->(target:Event)
        WHERE source.id IN $event_ids AND target.id IN $event_ids
        AND type(r) IN {rel_types_str}
        RETURN source.id AS source_id,
            source.name AS source_name,
            target.id AS target_id,
            target.name AS target_name,
            coalesce(r.confidence,0.0) AS confidence,
            r.reason AS causality_reason,
            type(r) AS rel_type
        ORDER BY confidence DESC
        """
        return self.execute_query(cypher, {"event_ids": event_ids})


    def create_plot_node(self, plot_data: Dict[str, Any]) -> bool:
        """
        创建 Plot 节点
        
        Args:
            plot_data: Plot 数据字典
                必须包含：
                - id: Plot ID
                - name: Plot 名称（原 title）
                - description: Plot 描述（原 summary）
                - main_characters, locations, time, reason: 其他属性
        
        Returns:
            bool: 创建是否成功
        """
        cypher = """
        MERGE (p:Plot {id: $plot_id})
        SET p.name = $name,
            p.description = $description,
            p.properties = $properties
        RETURN p.id AS plot_id
        """
        
        # 统一收集附加属性到 properties
        properties = {
            "main_characters": plot_data.get("main_characters"),
            "locations": plot_data.get("locations"),
            "time": plot_data.get("time"),
            "reason": plot_data.get("reason"),
            "related_events": plot_data.get("event_ids", []),
            "event_chain": "->".join(plot_data.get("event_ids", [])),
            "theme": plot_data.get("theme", ""),
            "goal": plot_data.get("goal", ""),
            "conflict": plot_data.get("conflict", ""),
            "resolution": plot_data.get("resolution", ""),
        }
        
        params = {
            "plot_id": plot_data["id"],
            "name": plot_data["title"],  # 原 title
            "description": plot_data["summary"],  # 原 summary
            "properties": json.dumps(properties, ensure_ascii=False)
        }
        
        try:
            result = self.execute_query(cypher, params)
            return len(list(result)) > 0
        except Exception as e:
            print(f"创建 Plot 节点失败: {e}")
            return False


    def create_plot_event_relationships(self, plot_id: str, event_ids: List[str]) -> bool:
        """
        创建 HAS_EVENT 关系，并写入中文含义 relation_name=“包含事件”
        """
        import hashlib

        rel_data = []
        for event_id in event_ids:
            rel_id = "rel_" + hashlib.sha1(f"{plot_id}-HAS_EVENT-{event_id}".encode("utf-8")).hexdigest()[:16]
            rel_data.append({
                "src_id": plot_id,
                "tgt_id": event_id,
                "rel_id": rel_id,
                "predicate": "HAS_EVENT",
                "relation_name": "包含事件",
            })

        cypher = """
        UNWIND $data AS row
        MATCH (p:Plot {id: row.src_id})
        MATCH (e:Event {id: row.tgt_id})
        MERGE (p)-[r:HAS_EVENT {id: row.rel_id}]->(e)
        SET r.predicate     = row.predicate,
            r.relation_name = row.relation_name
        RETURN count(r) AS relationships_created
        """
        try:
            result = self.execute_query(cypher, {"data": rel_data})
            count = list(result)[0]['relationships_created']
            return count == len(event_ids)
        except Exception as e:
            print(f"创建 HAS_EVENT 关系失败: {e}")
            return False

    
    def create_plot_relations(self, edges: List[Dict[str, Any]]) -> bool:
        """
        批量创建情节关系（最终版，含中文 relation_name）：
        - 有向：PLOT_PREREQUISITE_FOR(“前置/铺垫”) / PLOT_ADVANCES(“推进”) /
                PLOT_BLOCKS(“阻碍”) / PLOT_RESOLVES(“解决”)
        - 无向：PLOT_CONFLICTS_WITH(“冲突”) / PLOT_PARALLELS(“平行/呼应”)

        edges: [{"src","tgt","relation_type","confidence","reason"}, ...]
        """
        if not edges:
            print("[!] 没有传入任何情节关系，跳过创建。")
            return False

        import hashlib

        DIRECTED = {
            "PLOT_PREREQUISITE_FOR",
            "PLOT_ADVANCES",
            "PLOT_BLOCKS",
            "PLOT_RESOLVES",
        }
        UNDIRECTED = {
            "PLOT_CONFLICTS_WITH",
            "PLOT_PARALLELS",
        }
        ALLOWED = DIRECTED | UNDIRECTED
        NAME_ZH = {
            "PLOT_PREREQUISITE_FOR": "前置/铺垫",
            "PLOT_ADVANCES": "推进",
            "PLOT_BLOCKS": "阻碍",
            "PLOT_RESOLVES": "解决",
            "PLOT_CONFLICTS_WITH": "冲突",
            "PLOT_PARALLELS": "平行/呼应",
        }

        norm: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for e in edges:
            rtype = e.get("relation_type")
            if rtype not in ALLOWED:
                continue
            src, tgt = e.get("src"), e.get("tgt")
            if not src or not tgt or src == tgt:
                continue

            # 无向关系规范化：只存 (min, max)
            if rtype in UNDIRECTED and tgt < src:
                src, tgt = tgt, src

            key = (src, rtype, tgt)
            if key in seen:
                continue
            seen.add(key)

            rel_id = "rel_" + hashlib.sha1(f"{src}|{rtype}|{tgt}".encode("utf-8")).hexdigest()[:16]
            norm.append({
                "src_id": src,
                "tgt_id": tgt,
                "rel_id": rel_id,
                "predicate": rtype,
                "relation_name": NAME_ZH[rtype],
                "confidence": float(e.get("confidence") or 0.0),
                "reason": e.get("reason", ""),
            })

        if not norm:
            print("[!] 过滤后无可写入的情节关系。")
            return False

        all_created = True
        for rtype in sorted({e["predicate"] for e in norm}):
            subset = [e for e in norm if e["predicate"] == rtype]
            cypher = f"""
            UNWIND $data AS row
            MATCH (p1:Plot {{id: row.src_id}})
            MATCH (p2:Plot {{id: row.tgt_id}})
            MERGE (p1)-[r:{rtype} {{id: row.rel_id}}]->(p2)
            SET r.predicate     = row.predicate,
                r.relation_name = row.relation_name,
                r.confidence    = row.confidence,
                r.reason        = row.reason
            RETURN count(r) AS relationships_created
            """
            try:
                result = self.execute_query(cypher, {"data": subset})
                created = list(result)[0]["relationships_created"]
                if created != len(subset):
                    all_created = False
                    print(f"[!] {rtype} 仅创建 {created}/{len(subset)} 条，可能存在节点缺失或并发竞争。")
                else:
                    print(f"[✓] {rtype} 已创建 {created} 条关系")
            except Exception as e:
                print(f"[❌] 创建 {rtype} 关系失败: {e}")
                all_created = False

        return all_created

        
    def create_event_plot_graph(self):
        """
        用白名单关系创建 Event-Plot 专用 GDS 图：
        - 事件三类：EVENT_CAUSES / EVENT_INDIRECT_CAUSES / EVENT_PART_OF
        - Plot 六类：PLOT_PREREQUISITE_FOR / PLOT_ADVANCES / PLOT_BLOCKS / PLOT_RESOLVES / PLOT_CONFLICTS_WITH / PLOT_PARALLELS
        - HAS_EVENT
        """
        # 先删旧图
        try:
            self.execute_query("CALL gds.graph.drop('event_plot_graph', false)")
        except Exception:
            pass

        allowed = [
            "EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF",
            "HAS_EVENT",
            "PLOT_PREREQUISITE_FOR","PLOT_ADVANCES","PLOT_BLOCKS","PLOT_RESOLVES",
            "PLOT_CONFLICTS_WITH","PLOT_PARALLELS"
        ]
        types_str = ", ".join(f"\"{t}\"" for t in allowed)

        cypher = f"""
        CALL gds.graph.project.cypher(
        'event_plot_graph',
        'MATCH (n) RETURN id(n) AS id',
        'MATCH (a)-[r]->(b)
        WHERE type(r) IN [{types_str}]
        RETURN id(a) AS source, id(b) AS target'
        );
        """
        self.execute_query(cypher)
        print("✅ 创建 Event Plot Graph（事件因果 + HAS_EVENT + 6 类 Plot 边）")




    def write_plot_to_neo4j(self, plot_data: Dict[str, Any]) -> bool:
        """
        完整的Plot写入功能
        
        Args:
            plot_data: Plot数据字典，包含id、title、summary、event_ids、structure
            
        Returns:
            bool: 写入是否成功
        """
        try:
            # 1. 创建Plot节点
            if not self.create_plot_node(plot_data):
                return False
            
            # 2. 创建HAS_EVENT关系
            event_ids = plot_data.get("event_ids", [])
            if event_ids and not self.create_plot_event_relationships(plot_data["id"], event_ids):
                return False
            
            # print(f"成功写入Plot: {plot_data['id']}")
            return True
            
        except Exception as e:
            print(f"写入Plot到Neo4j失败: {e}")
            return False
    
    
    def load_connected_components_subgraph(self, node_ids: List[int]) -> tuple[Dict[int, Dict], List[Dict]]:
        """
        从 Neo4j 加载一个 CC 的所有节点和边
        
        Args:
            node_ids: Neo4j 内部节点 ID 列表

        Returns:
            - node_map: {nodeId -> 属性字典}
            - edges: List of {sid, tid, w, reason}
        """
        # 1. 节点
        cypher_nodes = f"""
        UNWIND $ids AS nid
        MATCH (n) WHERE n.id = nid
        RETURN n.id AS dbid,
                n.id AS eid,
                n.embedding AS emb
        """
        nodes = self.execute_query(cypher_nodes, {"ids": node_ids})
        node_map = {n["dbid"]: n for n in nodes}

        # 2. 边
        cypher_edges = """
        MATCH (u:Event)-[r]->(v:Event)
        WHERE u.id IN $ids AND v.id IN $ids AND type(r) IN ["EVENT_CAUSES","EVENT_INDIRECT_CAUSES","EVENT_PART_OF"]
        RETURN u.id AS sid,
            v.id AS tid,
            coalesce(r.confidence,0.0) AS confidence,
            r.reason AS reason
        """

        edges = self.execute_query(cypher_edges, {"ids": node_ids})
        return node_map, edges
    
    
    def fetch_scc_components(self, graph_name, min_size: int = 0) -> List[List[int]]:
        """
        调用 GDS 的 scc.stream 返回强连通体
        针对 size>1 的组件才需要断环
        """
        cypher = f"""
        CALL gds.scc.stream('{graph_name}')
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS n, componentId
        RETURN componentId,
            collect(n.id) AS nodeIds
        """
        sccs = self.execute_query(cypher)
        sccs = [c["nodeIds"] for c in sccs if len(c["nodeIds"]) >= min_size]
        # print(f"Detected { len(sccs)} SCCs with size>1")
        return sccs

    def fetch_wcc_components(self, graph_name, min_size: int = 0) -> List[List[int]]:
        """
        调用 GDS 的 scc.stream 返回强连通体
        针对 size>1 的组件才需要断环
        """
        cypher = f"""
        CALL gds.wcc.stream('{graph_name}')
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS n, componentId
        RETURN componentId,
            collect(n.id) AS nodeIds
        """
        sccs = self.execute_query(cypher)
        sccs = [c["nodeIds"] for c in sccs if len(c["nodeIds"]) >= min_size]
        # print(f"Detected { len(sccs)} WCCs with size>1")
        return sccs


    def get_plot_statistics(self) -> Dict[str, int]:
        """
        获取Plot图谱统计信息
        
        Returns:
            Dict[str, int]: 统计信息
        """
        cypher = f"""
        MATCH (p:Plot)
        OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (s:{self.meta['section_label']})-[:{self.meta['contains_pred']}]->(e)
        RETURN count(DISTINCT p) AS plot_count,
               count(DISTINCT e) AS event_count,
               count(DISTINCT s) AS section_count
        """
        
        result = self.execute_query(cypher)
        return dict(list(result)[0])
    
    def get_starting_events(self):
        cypher = """
        MATCH (e:Event)
        WHERE NOT ()-[:EVENT_CAUSES|:EVENT_INDIRECT_CAUSES|:EVENT_PART_OF]->(e)
        RETURN e.id AS event_id
        """

        result = self.execute_query(cypher)
        result = [e["event_id"] for e in result]
        return result
    
    def create_plot_event_relationships(self, plot_id: str, event_ids: List[str]) -> bool:
        """
        创建 HAS_EVENT 关系，并写入中文含义 relation_name=“包含事件”
        """
        import hashlib

        rel_data = []
        for event_id in event_ids:
            rel_id = "rel_" + hashlib.sha1(f"{plot_id}-HAS_EVENT-{event_id}".encode("utf-8")).hexdigest()[:16]
            rel_data.append({
                "src_id": plot_id,
                "tgt_id": event_id,
                "rel_id": rel_id,
                "predicate": "HAS_EVENT",
                "relation_name": "包含事件",
            })

        cypher = """
        UNWIND $data AS row
        MATCH (p:Plot {id: row.src_id})
        MATCH (e:Event {id: row.tgt_id})
        MERGE (p)-[r:HAS_EVENT {id: row.rel_id}]->(e)
        SET r.predicate     = row.predicate,
            r.relation_name = row.relation_name
        RETURN count(r) AS relationships_created
        """
        try:
            result = self.execute_query(cypher, {"data": rel_data})
            count = list(result)[0]['relationships_created']
            return count == len(event_ids)
        except Exception as e:
            print(f"创建 HAS_EVENT 关系失败: {e}")
            return False



    
    def find_event_chain(self, entity_id: str, min_confidence: float = 0.0):
        """
        从指定起点事件出发，返回所有到“终点事件”的路径。
        终点事件定义：在所考虑的关系类型中不再有出边。
        仅保留满足 confidence 阈值的边。

        考虑的关系类型（含历史兼容别名）：
        - EVENT_CAUSES / EVENT_CAUSE
        - EVENT_INDIRECT_CAUSE / EVENT_INDIRECT_CAUSES
        - EVENT_PART_OF
        """
        # 关系类型集合（含旧名，确保兼容）
        rel_types = [
            "EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"
        ]
        rel_types_str = "|".join(rel_types)

        cypher = f"""
        MATCH path = (start:Event {{id: $entity_id}})-[
            r:{rel_types_str}*
        0..]->(end:Event)
        WHERE
        // 路径上所有关系满足置信度阈值
        ALL(rel IN relationships(path)
            WHERE coalesce(rel.confidence, 0.0) >= $min_confidence)
        // 终点：在所考虑的关系集合中不再有出边
        AND NOT (end)-[:{rel_types_str}]->()
        RETURN [n IN nodes(path) | n.id] AS event_chain
        """

        results = self.execute_query(
            cypher,
            {
                "entity_id": entity_id,
                "min_confidence": float(min_confidence)
            }
        )
        return [record["event_chain"] for record in results if "event_chain" in record]

    
    def reset_event_plot_graph(self):
        cypher = """
        MATCH ()-[r]->()
        WHERE type(r) IN [
        "HAS_EVENT",
        "PLOT_PREREQUISITE_FOR","PLOT_ADVANCES","PLOT_BLOCKS","PLOT_RESOLVES",
        "PLOT_CONFLICTS_WITH","PLOT_PARALLELS",
        "PLOT_CONTRIBUTES_TO","PLOT_SETS_UP" // 历史兼容可留
        ]
        DELETE r
        """
        self.execute_query(cypher)
        
        cypher = """
        MATCH (p:Plot)
        DETACH DELETE p;
        """
        self.execute_query(cypher)
        print("✅ Event Plot Graph已重置")
    
    
    def get_plot_pairs(self, threshold=0):
        """
        召回候选情节对（返回字典而非元组），并带回最短路径长度：
        - 仅沿以下边联通：事件三类 + HAS_EVENT + 六类 Plot 边
        - 优先选择路径更短的情节对
        - 总量不超过 3 × Plot 数量
        - 二次过滤：文本向量相似度 & node2vec 图相似度

        返回: List[Dict]，每项形如 {"src": str, "tgt": str, "path_len": int}
        """
        # 1) 只在白名单关系上找 1..5 跳内最短路径
        cypher = """
        MATCH (p1:Plot), (p2:Plot)
        WHERE id(p1) < id(p2)
        MATCH path = (p1)-[*1..5]-(p2)
        WITH p1, p2, min(length(path)) AS path_len
        RETURN p1.id AS src, p2.id AS tgt, path_len
        """
        results = self.execute_query(cypher)

        # 2) 计算 Plot 数 & 设上限（建议≈ 3x）
        plot_cypher = "MATCH (p:Plot) RETURN count(DISTINCT p) AS plot_count"
        res = self.execute_query(plot_cypher)
        num_plots = int(res[0]["plot_count"]) if res else 0
        max_num_relations = num_plots * 3

        # 3) 按路径长度分桶
        pair_maps: Dict[int, List[Dict[str, Any]]] = {}
        for row in results:
            d = int(row["path_len"])
            item = {"src": row["src"], "tgt": row["tgt"], "path_len": d}
            pair_maps.setdefault(d, []).append(item)

        # 4) 依次从短到长选取，直到达到上限
        import random
        selected_pairs: List[Dict[str, Any]] = []
        count = 0
        for distance in sorted(pair_maps.keys()):
            bucket = pair_maps[distance]
            remain = max_num_relations - count
            if remain <= 0:
                break
            if len(bucket) <= remain:
                selected_pairs.extend(bucket)
                count += len(bucket)
            else:
                selected_pairs.extend(random.sample(bucket, remain))
                count += remain
                break

        # 5) 相似度过滤（文本 + 图 node2vec）
        if threshold > 0:
            filtered: List[Dict[str, Any]] = []
            for item in selected_pairs:
                src, tgt = item["src"], item["tgt"]
                # 这两个函数返回 None 时跳过该对
                sim = self.compute_semantic_similarity(src, tgt)
                gsim = self.compute_graph_similarity(src, tgt, "node2vecEmbedding")
                if sim is None or gsim is None:
                    continue
                if sim >= threshold and gsim >= threshold:
                    filtered.append(item)
        else: 
            filtered = selected_pairs

        return filtered

    
    
    def create_event_plot_graph(self):
        cypher = """
        CALL gds.graph.drop('event_plot_graph', false);
        """
        self.execute_query(cypher) # 删除已有的图
        
        cypher = """
        CALL gds.graph.project(
        'event_plot_graph',
        {
            Plot: { properties: ['embedding'] },
            Event: { properties: ['embedding'] },
            Character: { properties: ['embedding'] },
            Location: { properties: ['embedding'] },
            Concept: { properties: ['embedding'] },
            Object: { properties: ['embedding'] }
        },
        '*'
        );
        """
        self.execute_query(cypher)
        print("✅ 创建 Event Plot Graph")
        
    def run_node2vec(self):
        cypher = """
        CALL gds.node2vec.write(
        'event_plot_graph',
        {
            embeddingDimension: 128,        // 向量维度
            walkLength: 80,                  // 每条游走路径长度
            walksPerNode: 20,                  // 每个节点起点的游走次数
            inOutFactor: 1.0,                 // p 参数（回访概率）
            returnFactor: 1.0,                // q 参数（前进概率）
            concurrency: 4,                   // 并行线程数
            writeProperty: 'node2vecEmbedding' // 写回属性名
        }
        )
        YIELD nodeCount, nodePropertiesWritten;
        """
        self.execute_query(cypher)
        print("✅ 创建 Node2Vec向量至属性 node2vecEmbedding")
        