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
        relation_type: Optional[str] = None,
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
        if relation_type:
            params["rel_type"] = relation_type
        if predicate:
            params["predicate"] = predicate
        if entity_types:
            params["etypes"] = entity_types

        # entity type 过滤语句
        type_filter = "AND target.type IN $etypes" if entity_types else ""
        pred_filter = "AND rel.predicate = $predicate" if predicate else ""
        rel_type_clause = f":{relation_type}" if relation_type else ""

        results = []

        with self.driver.session() as session:
            # 正向关系
            forward_cypher = f"""
            MATCH (source)-[rel{rel_type_clause }]->(target)
            WHERE source.id = $source_id
              AND rel.predicate IS NOT NULL
              {pred_filter}
              {type_filter}
            RETURN target, rel
            LIMIT $limit
            """
            # print("[CHECK] forward_cypher: ", session.run(forward_cypher, params))

            for record in session.run(forward_cypher, params):
                entity, relation = self._process_entity_relation_record(record, source_id, "forward")
                if return_relations:
                    results.append((entity, relation))
                else:
                    results.append(entity)

            # 反向关系
            backward_cypher = f"""
            MATCH (target)-[rel{rel_type_clause}]->(source)
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
    
    def process_all_embeddings(self, exclude_node_types: List[str] = [], exclude_rel_types: List[str] = []):
        """
        自动处理所有节点标签和所有边，为其生成 embedding 并写回图数据库。
        节点 embedding 输入：name + description (+ properties)
        边 embedding 输入：properties.description
        """
        # === 获取所有实体类型（标签） ===
        node_types = self.list_entity_types()

        # === 处理节点嵌入 ===
        print("🚀 开始处理节点嵌入...")
        for node in exclude_node_types:
            if node in node_types:
                node_types.remove(node)
                
        print(f"📌 实体类型标签: {node_types}")
        nodes = self.fetch_all_nodes(node_types)
        for n in  tqdm(nodes, desc="Encoding Nodes", ncols=80):
            try:
                emb = self.encode_node_embedding(n)
                self.update_node_embedding(n["id"], emb)
            except Exception as e:
                print(f"⚠️ Node {n.get('id')} embedding failed:", str(e))

        print(f"✅ 节点嵌入完成，共处理 {len(nodes)} 个节点")

        # === 处理关系嵌入 ===
        print("🚀 开始处理边嵌入...")
        rel_types = self.list_relationship_types()
        for rel in exclude_rel_types: # 移除不需要考虑的边关系
            if rel in rel_types:
                rel_types.remove(rel)
        
        rels = self.fetch_all_relations(rel_types)
        
        for r in tqdm(rels, desc="Encoding Edges", ncols=80):
            try:
                emb = self.encode_relation_embedding(r)
                if emb:
                    self.update_relation_embedding(r["id"], emb)
            except Exception as e:
                print(f"⚠️ Relation {r.get('id')} embedding failed:", str(e))

        print(f"✅ 边嵌入完成，共处理 {len(rels)} 条边")
        
        
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
    
    
    def create_subgraph(
        self,
        graph_name: str = "subgraph_1",
        exclude_node_labels: Optional[List[str]] = None,
        exclude_rel_types: Optional[List[str]] = None,
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

        exclude_node_labels = exclude_node_labels or ["Scene"]
        exclude_rel_types   = exclude_rel_types   or ["SCENE_CONTAINS"]

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
            label_filter = " AND ".join([f"NOT '{lbl}' IN labels(n)" for lbl in exclude_node_labels]) or "true"
            node_query = f"""
            MATCH (n) WHERE {label_filter}
            RETURN id(n) AS id
            """

            #   关系：排除指定类型 & 排除与被排除节点相连的边
            rel_filter = " AND ".join([f"type(r) <> '{rt}'" for rt in exclude_rel_types]) or "true"
            # 额外保证两端节点都不是被排除标签
            node_label_neg = " AND ".join([f"NOT '{lbl}' IN labels(a)" for lbl in exclude_node_labels] +
                                        [f"NOT '{lbl}' IN labels(b)" for lbl in exclude_node_labels]) or "true"

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

            print(f"[+] 已创建 GDS 子图 {graph_name}（排除标签 {exclude_node_labels}，排除边 {exclude_rel_types}）")

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
        max_depth: int = 3,
        max_pairs: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        返回同社区 & 路径在 max_depth 内可达的事件对 ID 列表
        """
        q = f"""
        MATCH (e1:Event)
        MATCH (e2:Event)
        WHERE e1.community = e2.community AND id(e1) < id(e2)
          AND EXISTS {{
              MATCH p = (e1)-[*1..{max_depth}]-(e2)
              WHERE ALL(r IN relationships(p) WHERE type(r) <> 'SCENE_CONTAINS')
          }}
        RETURN e1.id AS srcId, e2.id AS dstId
        """ + (f"LIMIT {max_pairs}" if max_pairs else "")
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

    def get_event_details(self, event_ids: List[str]) -> List[Dict[str, Any]]:
        """
        获取事件详细信息
        
        Args:
            event_ids: 事件ID列表
            
        Returns:
            List[Dict]: 事件详细信息列表
        """
        cypher = """
        MATCH (e:Event)
        WHERE e.id IN $event_ids
        OPTIONAL MATCH (s:Scene)-[:SCENE_CONTAINS]->(e)
        RETURN e.id as event_id,
            e.name as event_name,
            e.description as event_description,
            e.participants as participants,
            e.location as location,
            e.time as time,
            collect(DISTINCT s.id) as scene_ids,
            collect(DISTINCT s.name) as scene_names
        """
        
        params = {"event_ids": event_ids}
        result = self.execute_query(cypher, params)
        return [dict(record) for record in result]

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

    def get_plot_statistics(self) -> Dict[str, int]:
        """
        获取Plot图谱统计信息
        
        Returns:
            Dict[str, int]: 统计信息
        """
        cypher = """
        MATCH (p:Plot)
        OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (s:Scene)-[:SCENE_CONTAINS]->(e)
        RETURN count(DISTINCT p) as plot_count,
            count(DISTINCT e) as event_count,
            count(DISTINCT s) as scene_count
        """
        
        result = self.execute_query(cypher)
        return dict(list(result)[0])
    
    