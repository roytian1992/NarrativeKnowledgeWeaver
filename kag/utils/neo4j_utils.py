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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# from kag.builder.kg_builder_2 import DOC_TYPE_META


DOC_TYPE_META: Dict[str, Dict[str, str]] = {
    "screenplay": {
        "section_label": "Scene",
        "title": "scene_name",
        "subtitle_key": "sub_scene_name",
        "contains_pred": "SCENE_CONTAINS",
    },
    "novel": {
        "section_label": "Chapter",
        "title": "chapter_name",
        "subtitle": "sub_chapter_name",
        "contains_pred": "CHAPTER_CONTAINS",
    },
}

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
        self.dim = 768
        
    def load_emebdding_model(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print("向量模型已加载")
    
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
        type_filter = "AND target.type IN $etypes" if entity_types else ""
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
        if relation_type == "EVENT_CAUSES":
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
            text = f"{name}：{desc}。{prop_text}"
        else:
            text = f"{name}：{desc}"
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def encode_relation_embedding(self, rel: Dict) -> Optional[List[float]]:
        try:
            props = rel.get("properties", "")
            props_dict = json.loads(props) if isinstance(props, str) else props
            desc = props_dict.get("description", "")
            if desc:
                return self.model.encode(desc, normalize_embeddings=True).tolist()
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
    
    def process_all_embeddings(self, exclude_entity_types: List[str] = [], exclude_relation_types: List[str] = []):
        """
        自动处理所有节点标签和所有边，为其生成 embedding 并写回图数据库。
        节点 embedding 输入：name + description (+ properties)
        边 embedding 输入：properties.description
        """
        # === 获取所有实体类型（标签） ===
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
        embed = self.model.encode(text, normalize_embeddings=normalize).tolist()
        return self._query_entity_knn(embed, top_k=top_k)
    
    
    def compute_semantic_similarity(self, node_id_1, node_id_2):
        query = f"""
        MATCH (a {{id: '{node_id_1}'}}), (b {{id: '{node_id_2}'}})                                          
        RETURN gds.similarity.cosine(a.embedding, b.embedding) AS similarity
        """
        result = self.execute_query(query)
        return result[0].get("similarity")
    
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


    def create_event_causality_graph(self, graph_name: str = "event_causality_graph", force_refresh: bool = True):
        """
        创建一个只包含 Event 节点 + EVENT_CAUSES 边的 GDS 图，用于因果分析
        """
        with self.driver.session() as s:
            if force_refresh:
                s.run("CALL gds.graph.drop($name, false) YIELD graphName", name=graph_name)
                print(f"[✓] 已删除旧图 {graph_name}")

            s.run("""
            CALL gds.graph.project(
                $name,
                'Event',
                {
                    EVENT_CAUSES: {
                        orientation: 'NATURAL',
                        properties: ['weight']
                    }
                }
            )
            """, name=graph_name)

            print(f"[+] 已创建因果子图 {graph_name}（仅包含 Event 节点与 EVENT_CAUSES 边）")
            
            result = s.run("""
                MATCH (:Event)-[r:EVENT_CAUSES]->(:Event)
                RETURN count(r) AS edge_count
            """)
            edge_count = result.single()["edge_count"]
            print(f"[✓] 当前 EVENT_CAUSES 边数量：{edge_count}")

    
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
        rows: [{srcId, dstId, weight, reason}]
        """
        if not rows:
            return
        self.execute_query("""
        UNWIND $rows AS row
        MATCH (s:Event {id: row.srcId})
        MATCH (t:Event {id: row.dstId})
        MERGE (s)-[ca:EVENT_CAUSES]->(t)
        SET ca.weight = row.weight,
            ca.reason = row.reason,
            ca.confidence = row.confidence,
            ca.predicate = row.predicate
        """, {"rows": rows})
        print(f"[+] 已写入/更新 EVENT_CAUSES 关系 {len(rows)} 条")
    
    def get_all_events_with_causality(self) -> List[Dict[str, Any]]:
        """
        获取所有事件及其因果关系信息
        
        Returns:
            List[Dict]: 包含事件ID、属性和因果关系的列表
        """
        cypher = """
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[r:EVENT_CAUSES]->(target:Event)
        OPTIONAL MATCH (source:Event)-[r2:EVENT_CAUSES]->(e)
        RETURN e.id as event_id, 
            e.name as event_name,
            e.description as event_description,
            e.participants as participants,
            collect(DISTINCT {target: target.id, weight: r.weight}) as outgoing_causes,
            collect(DISTINCT {source: source.id, weight: r2.weight}) as incoming_causes
        """
        
        result = self.execute_query(cypher)
        return [dict(record) for record in result]

    def get_causality_edges_by_weight(self, threshold: str = "Medium") -> List[Dict[str, Any]]:
        """
        根据权重阈值获取因果关系边
        
        Args:
            threshold: 权重阈值 ("High", "Medium", "Low")
            
        Returns:
            List[Dict]: 因果关系边列表
        """
        # 定义权重映射
        weight_hierarchy = {
            "High": 1.0,
            "Medium": 0.6, 
            "Low": 0.3
        }
        
        weight_threshold = weight_hierarchy.get(threshold, 0.6)
        
        cypher = """
        MATCH (source:Event)-[r:EVENT_CAUSES]->(target:Event)
        WHERE r.weight >= $weight_threshold
        RETURN source.id AS source_id, 
            target.id AS target_id, 
            r.weight AS weight
        """
        
        params = {"weight_threshold": weight_threshold}
        result = self.execute_query(cypher, params)
        return [dict(record) for record in result]

    def identify_event_clusters_by_connectivity(self, threshold: str = "Medium") -> List[List[str]]:
        """
        使用GDS连通分量算法识别事件聚类
        
        Args:
            threshold: 因果关系权重阈值
            
        Returns:
            List[List[str]]: 事件聚类列表，每个聚类包含事件ID列表
        """
        # 1. 创建基于权重阈值的投影图
        graph_name = f"event_causality_graph_{threshold.lower()}"
        
        # 删除可能存在的旧图
        drop_cypher = f"CALL gds.graph.drop('{graph_name}') YIELD graphName"
        try:
            self.execute_query(drop_cypher)
        except:
            pass  # 图不存在时忽略错误
        
        # 获取权重过滤条件
        weight_hierarchy = {
            "High": 1.0,
            "Medium": 0.6, 
            "Low": 0.3
        }
        weight_threshold = weight_hierarchy.get(threshold, 0.6)
        
        # 创建投影图 - 只包含满足权重条件的关系
        create_graph_cypher = f"""
        CALL gds.graph.project.cypher(
            '{graph_name}',
            'MATCH (n:Event) RETURN id(n) AS id',
            'MATCH (a:Event)-[r:EVENT_CAUSES]->(b:Event) 
            WHERE r.weight >= {weight_threshold}
            RETURN id(a) AS source, id(b) AS target, r.weight AS weight'
        )
        """
        # print("[CHECK] create_graph_cypher", create_graph_cypher)
        
        self.execute_query(create_graph_cypher)
        
        # 2. 运行连通分量算法
        wcc_cypher = f"""
        CALL gds.wcc.stream('{graph_name}')
        YIELD nodeId, componentId
        RETURN gds.util.asNode(nodeId).id as event_id, componentId
        ORDER BY componentId, event_id
        """
        
        result = self.execute_query(wcc_cypher)
        # print("[CHECK] result: ", result)
        
        # 3. 组织结果为聚类
        clusters = {}
        for record in result:
            component_id = record['componentId']
            event_id = record['event_id']
            
            if component_id not in clusters:
                clusters[component_id] = []
            clusters[component_id].append(event_id)
        
        # print("[CHECK] clusters: ", clusters)
        # 4. 清理图
        # self.execute_query(drop_cypher)
        
        # 5. 过滤聚类 - 只保留通过权重阈值连接的事件
        filtered_clusters = []
        edges = self.get_causality_edges_by_weight(threshold)
        # print("[CHECK] edges: ", edges)
        
        # 构建满足权重条件的连接图
        connected_events = set()
        for edge in edges:
            connected_events.add(edge['source_id'])
            connected_events.add(edge['target_id'])
        
        for cluster in clusters.values():
            # 只保留有满足权重条件连接的聚类，且聚类大小大于1
            if len(cluster) > 1:
                cluster_has_valid_connections = any(event_id in connected_events for event_id in cluster)
                if cluster_has_valid_connections:
                    filtered_clusters.append(cluster)
        
        return filtered_clusters
            

    def _fallback_clustering(self, threshold: str) -> List[List[str]]:
        """
        降级聚类方法：基于直接因果关系的简单聚类
        
        Args:
            threshold: 权重阈值
            
        Returns:
            List[List[str]]: 事件聚类列表
        """
        edges = self.get_causality_edges_by_weight(threshold)
        
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
        """
        获取事件间的因果路径
        
        Args:
            event_ids: 事件ID列表
            
        Returns:
            List[Dict]: 因果路径信息
        """
        cypher = """
        MATCH (source:Event)-[r:EVENT_CAUSES]->(target:Event)
        WHERE source.id IN $event_ids AND target.id IN $event_ids
        RETURN source.id as source_id,
            source.name as source_name,
            target.id as target_id,
            target.name as target_name,
            r.weight as weight,
            r.description as causality_description
        ORDER BY 
            CASE r.weight 
                WHEN 'High' THEN 1 
                WHEN 'Medium' THEN 2 
                WHEN 'Low' THEN 3 
                ELSE 4 
            END
        """
        
        params = {"event_ids": event_ids}
        result = self.execute_query(cypher, params)
        return [dict(record) for record in result]

    def create_plot_node(self, plot_data: Dict[str, Any]) -> bool:
        """
        创建Plot节点
        
        Args:
            plot_data: Plot数据字典
            
        Returns:
            bool: 创建是否成功
        """
        cypher = """
        CREATE (p:Plot {
            id: $plot_id,
            title: $title,
            summary: $summary,
            structure_type: $structure_type,
            narrative_roles: $narrative_roles,
            created_at: datetime()
        })
        RETURN p.id as plot_id
        """
        
        params = {
            "plot_id": plot_data["id"],
            "title": plot_data["title"],
            "summary": plot_data["summary"],
            "structure_type": plot_data.get("structure", {}).get("type", "起承转合"),
            "narrative_roles": str(plot_data.get("structure", {}).get("narrative_roles", {}))
        }
        
        try:
            result = self.execute_query(cypher, params)
            return len(list(result)) > 0
        except Exception as e:
            print(f"创建Plot节点失败: {e}")
            return False

    def create_has_event_relationships(self, plot_id: str, event_ids: List[str]) -> bool:
        """
        创建HAS_EVENT关系
        
        Args:
            plot_id: Plot ID
            event_ids: 事件ID列表
            
        Returns:
            bool: 创建是否成功
        """
        cypher = """
        MATCH (p:Plot {id: $plot_id})
        MATCH (e:Event)
        WHERE e.id IN $event_ids
        CREATE (p)-[:HAS_EVENT]->(e)
        RETURN count(*) as relationships_created
        """
        
        params = {
            "plot_id": plot_id,
            "event_ids": event_ids
        }
        
        try:
            result = self.execute_query(cypher, params)
            count = list(result)[0]['relationships_created']
            return count == len(event_ids)
        except Exception as e:
            print(f"创建HAS_EVENT关系失败: {e}")
            return False

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
            if event_ids and not self.create_has_event_relationships(plot_data["id"], event_ids):
                return False
            
            print(f"成功写入Plot: {plot_data['id']}")
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
        MATCH (u)-[r:EVENT_CAUSES]->(v)
        WHERE u.id IN $ids AND v.id IN $ids
        RETURN u.id AS sid,
                v.id AS tid,
                r.weight AS weight,
                r.reason AS reason,
                r.confidence AS confidence
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
        WHERE NOT ()-[:EVENT_CAUSES]->(e)
        RETURN e.id AS event_id
        """
        result = self.execute_query(cypher)
        result = [e["event_id"] for e in result]
        return result
    
    def find_event_chain(self, entity_id: str, graph_name: str) -> List[List[str]]:
        """
        使用 GDS 的 DFS，从指定 entity_id 出发，在给定图中搜索所有因果路径（事件链）
        
        Args:
            entity_id: 事件节点 ID（如 'entity_123456'）
            graph_name: 已创建的 GDS 图名（如 'eventCausalGraph'）

        Returns:
            所有 DFS 路径构成的事件链列表，每条链是 event_id 的有序列表
        """
        cypher = """
        MATCH (e:Event {id: $entity_id})
        WITH e AS start_node
        CALL gds.dfs.stream($graph_name, { sourceNode: start_node })
        YIELD nodeIds
        RETURN [nodeId IN nodeIds | gds.util.asNode(nodeId).id] AS event_chain
        """
        
        results = self.execute_query(cypher, {"entity_id": entity_id, "graph_name": graph_name})
        return [record["event_chain"] for record in results if "event_chain" in record]

    