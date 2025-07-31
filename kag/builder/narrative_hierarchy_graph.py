"""
事件因果图构建器
负责构建事件因果关系的有向带权图和情节单元图谱
"""

import json
import pickle
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from pathlib import Path
from kag.llm.llm_manager import LLMManager
from kag.utils.neo4j_utils import Neo4jUtils
from kag.models.entities import Entity
from kag.builder.extractor import InformationExtractor
from ..storage.graph_store import GraphStore
from kag.utils.prompt_loader import PromptLoader
from kag.functions.regular_functions.plot_generation import PlotGenerator
import logging
import os


class EventCausalityBuilder:
    """
    事件因果图构建器
    
    主要功能：
    1. 从Neo4j加载和排序事件
    2. 通过连通体和社区过滤事件对
    3. 使用extractor检查因果关系
    4. 构建有向带权NetworkX图
    5. 保存和加载图数据
    6. 构建Plot情节单元图谱
    """
    
    def __init__(self, config):
        """
        初始化事件因果图构建器
        
        Args:
            config: KAG配置对象
        """
        self.config = config
        self.llm_manager = LLMManager(config)
        self.llm = self.llm_manager.get_llm()
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver)
        self.extractor = InformationExtractor(config, self.llm)
        self.event_fallback = [] # 可以加入Goal和Action

        # 初始化Plot相关组件
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
       
        self.prompt_loader = PromptLoader(prompt_dir)
        self.plot_generator = PlotGenerator(self.prompt_loader, self.llm)
        
        # Plot构建配置参数（默认值）
        self.causality_threshold = "Medium"
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        
        self.logger = logging.getLogger(__name__)

        # 缓存数据
        self.load_abbreviations("kag/schema/settings_schema.json")
        
        self.sorted_scenes = []
        self.event_list = []
        self.event2scene_map = {}
        self.allowed_rels = []
        self.max_depth = 3
        
        # 因果关系强度到权重的映射
        self.causality_weight_map = {
            "High": 1.0,
            "Medium": 0.6,
            "Low": 0.3
        }
        
        self.logger.info("EventCausalityBuilder初始化完成")
    
    def load_abbreviations(self, path: str):
        """
        从JSON文件加载缩写列表，返回格式化后的文本（适合插入提示词）
        
        Args:
            path: 缩写文件路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            abbr = json.load(f)
        abbr_list = abbr.get("abbreviations", [])

        formatted = []
        for item in abbr_list:
            line = f"- **{item['abbr']}**: {item['full']}（{item['zh']}） - {item['description']}"
            formatted.append(line)
        self.abbreviation_info = "\n".join(formatted)
        
        print(f"✅ 已加载 {len(abbr_list)} 个缩写定义")
    
    def build_event_list(self) -> List[Entity]:
        """
        构建排序后的事件列表
        
        Returns:
            排序后的事件列表
        """
        print("🔍 开始构建事件列表...")
        
        # 1. 获取所有场景并排序
        scene_entities = self.neo4j_utils.search_entities_by_type(
            entity_type="Scene"
        )
        
        self.sorted_scenes = sorted(
            scene_entities,
            key=lambda e: (
                int(e.properties.get("scene_number", 0)),
                int(e.properties.get("sub_scene_number", 0))
            )
        )
        
        print(f"✅ 找到 {len(self.sorted_scenes)} 个场景")
        
        # 2. 从场景中提取事件
        event_list = []
        event2scene_map = {}
        
        for scene in tqdm(self.sorted_scenes, desc="提取场景中的事件"):
            # 优先查找事件
            results = self.neo4j_utils.search_related_entities(
                source_id=scene.id, 
                relation_type="SCENE_CONTAINS", 
                entity_types=["Event"], 
                return_relations=False
            )
            
            # 如果场景中没有事件，则用动作或者目标来填充
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=scene.id, 
                    relation_type="SCENE_CONTAINS", 
                    entity_types=self.event_fallback, 
                    return_relations=False
                )
            
            for result in results:
                if result.id not in event2scene_map:
                    event2scene_map[result.id] = scene.id
                    event_list.append(result)
        
        self.event_list = event_list
        self.event2scene_map = event2scene_map
        
        print(f"✅ 构建完成，共找到 {len(event_list)} 个事件")
        return event_list
    
    def get_event_info(self, event_id: str) -> str:
        """
        获取事件的详细信息，用于因果关系检查
        
        Args:
            event_id: 事件ID
            
        Returns:
            格式化的事件信息字符串
        """
        event_node = self.neo4j_utils.get_entity_by_id(event_id)
        entity_types = self.neo4j_utils.list_entity_types()
        results = self.neo4j_utils.search_related_entities(
            source_id=event_id, 
            return_relations=True
        )
        
        relevant_info = []
        for result in results:
            info = self._get_relation_info(result[1])
            if info:
                relevant_info.append(info)
                
        event_description = event_node.description or "无具体描述"
        
        context = (
            f"（{event_node.name}）：{event_description}\n"
            f"相关信息有：\n" + "\n".join(relevant_info)
        )
        return context
    
    def _get_relation_info(self, relation) -> Optional[str]:
        """
        获取关系信息的格式化字符串
        
        Args:
            relation: 关系对象
            
        Returns:
            格式化的关系信息，如果是SCENE_CONTAINS则返回None
        """
        if relation.predicate == "SCENE_CONTAINS":
            return None
            
        subject_id = relation.subject_id
        subject_name = self.neo4j_utils.get_entity_by_id(subject_id).name
        object_id = relation.object_id
        object_name = self.neo4j_utils.get_entity_by_id(object_id).name
        relation_name = relation.properties.get("relation_name", relation.predicate)
        description = relation.properties.get("description", "")
        
        return f"{subject_name}-{relation_name}->{object_name}: {description}"
    
    def filter_event_pairs_by_community(
        self,
        events: List[Entity],
        max_depth: int = 3
    ) -> List[Tuple[Entity, Entity]]:
        """
        利用 Neo4j 中 Louvain 结果直接筛选同社区且 max_depth 内可达的事件对
        """
        # 把事件 ID 做成集合，便于后面实体映射
        id2entity = {e.id: e for e in events}

        pairs = self.neo4j_utils.fetch_event_pairs_same_community()
        # print("[CHECK]: ", pairs)
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))

        print(f"[✓] 同社区事件对: {len(filtered_pairs)}")
        return filtered_pairs

    def write_event_cause_edges(self, causality_results):
        rows = []
        for (src_id, dst_id), res in causality_results.items():
            weight = self.causality_weight_map.get(res["causal"], 0.3)
            rows.append({
                "srcId": src_id,
                "dstId": dst_id,
                "weight": weight,
                "reason": res["reason"],
                "predicate": "EVENT_CAUSES"
            })
        self.neo4j_utils.write_event_causes(rows)

    
    def check_causality_batch(
        self, 
        pairs: List[Tuple[Entity, Entity]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        批量检查事件对的因果关系
        
        Args:
            pairs: 事件对列表
            
        Returns:
            事件对ID到因果关系结果的映射
        """
        print(f"🔍 开始批量检查 {len(pairs)} 对事件的因果关系...")
        
        causality_results = {}
        
        for src_event, tgt_event in tqdm(pairs, desc="检查因果关系"):
            # 获取事件信息
            event_1_info = self.get_event_info(src_event.id)
            event_2_info = self.get_event_info(tgt_event.id)
            
            # 调用extractor检查因果关系
            try:
                result_json_str = self.extractor.check_event_causality(
                    event_1_info, 
                    event_2_info, 
                    self.abbreviation_info
                )
                
                # 解析JSON结果
                result_dict = json.loads(result_json_str)
                
                # 存储结果，包括新的reverse字段
                pair_key = (src_event.id, tgt_event.id)
                causality_results[pair_key] = {
                    'src_event': src_event,
                    'tgt_event': tgt_event,
                    'causal': result_dict.get('causal', 'Low'),
                    'reason': result_dict.get('reason', ''),
                    'reverse': result_dict.get('reverse', False),  # 新增：是否反转因果方向
                    'raw_result': result_json_str
                }
                
            except Exception as e:
                print(f"⚠️ 检查事件对 {src_event.id} -> {tgt_event.id} 时出错: {e}")
                pair_key = (src_event.id, tgt_event.id)
                causality_results[pair_key] = {
                    'src_event': src_event,
                    'tgt_event': tgt_event,
                    'causal': 'Low',
                    'reason': f'检查过程出错: {e}',
                    'reverse': False,  # 出错时默认不反转
                    'raw_result': ''
                }
        
        print(f"✅ 因果关系检查完成")
        return causality_results
        
    def sort_event_pairs_by_scene_time(
        self,
        pairs: List[Tuple[Entity, Entity]]
    ) -> List[Tuple[Entity, Entity]]:
        """
        对事件对按照所属场景(scene_number, sub_scene_number)顺序排序，使早的事件排前面
        """
        def get_scene_order(event: Entity):
            scene_id = self.event2scene_map.get(event.id)
            if not scene_id:
                return (9999, 9999)  # 缺失信息排最后
            scene = self.neo4j_utils.get_entity_by_id(scene_id)
            if not scene:
                return (9999, 9999)
            return (
                int(scene.properties.get("scene_number", 0)),
                int(scene.properties.get("sub_scene_number", 0))
            )

        sorted_pairs = []
        for e1, e2 in pairs:
            if get_scene_order(e1) <= get_scene_order(e2):
                sorted_pairs.append((e1, e2))
            else:
                sorted_pairs.append((e2, e1))
        return sorted_pairs

    def initialize(self):
        # 1. 创建子图和计算社区划分
        self.neo4j_utils.delete_relation_type("EVENT_CAUSES")
        self.neo4j_utils.create_subgraph(
            graph_name="event_graph",
            exclude_node_labels=["Scene"],
            exclude_rel_types=["SCENE_CONTAINS", "EVENT_CAUSES"],
            force_refresh=True
        )

        self.neo4j_utils.run_louvain(
            graph_name="event_graph",
            write_property="community",
            force_run=True
        )
    
    def filter_pair_by_distance_and_similarity(self, pairs):
        filtered_pairs = []
        for pair in tqdm(pairs, desc="筛选节点对"):
            src_id, tgt_id = pair[0].id, pair[1].id
            reachable = self.neo4j_utils.check_nodes_reachable(src_id, tgt_id, excluded_rels=["SCENE_CONTAINS", "EVENT_CAUSES"])
            if reachable: # 如果节点间距离小于3，保留。
                filtered_pairs.append(pair)
            else:
                score = self.neo4j_utils.compute_semantic_similarity(src_id, tgt_id)
                if score >= 0.7: # 如果节点间的相似度大于等于0.7，保留。
                    filtered_pairs.append(pair)  
        return filtered_pairs
    
    def build_event_causality_graph(
        self,
        limit_events: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        完整的事件因果图构建流程
        
        Args:
            limit_events: 限制处理的事件数量（用于测试）
            
        Returns:
            构建完成的Neo4j有向图
        """
        print("🚀 开始完整的事件因果图构建流程...")
        
        # 2. 构建事件列表
        print("\n🔍 构建事件列表...")
        event_list = self.build_event_list()
        
        # 3. 限制事件数量（用于测试）
        if limit_events and limit_events < len(event_list):
            event_list = event_list[:limit_events]
            print(f"⚠️ 限制处理事件数量为: {limit_events}")
        
        # 4. 过滤事件对
        print("\n🔍 过滤事件对...")
        filtered_pairs = self.filter_event_pairs_by_community(event_list)
        filtered_pairs = self.filter_pair_by_distance_and_similarity(filtered_pairs)
        filtered_pairs = self.sort_event_pairs_by_scene_time(filtered_pairs)
        print("     最终候选事件对数量： ", len(filtered_pairs))
        # 5. 检查因果关系
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)
        
        # 6. 写回 EVENT_CAUSES
        print("\n🔗 写回 EVENT_CAUSES 关系...")
        self.write_event_cause_edges(causality_results)

    def build_plot_graph(self, verbose: bool = False) -> bool:
        """
        构建完整的Plot图谱
        
        Args:
            verbose: 是否输出详细日志
            
        Returns:
            bool: 构建是否成功
        """
        try:
            self.logger.info("开始构建Event-Plot-Scene图谱")
            
            # 1. 事件聚类 (Plot Candidates Identification)
            if verbose:
                print("🔍 步骤1: 识别Plot候选 (事件聚类)")
            
            event_clusters = self._identify_plot_candidates()
            
            if not event_clusters:
                self.logger.warning("未发现有效的事件聚类")
                return False
            
            self.logger.info(f"识别到 {len(event_clusters)} 个事件聚类")
            if verbose:
                for i, cluster in enumerate(event_clusters):
                    print(f"  聚类 {i+1}: {len(cluster)} 个事件 - {cluster}")
            
            # 2. 情节单元生成 (Plot Unit Construction)
            if verbose:
                print("🎭 步骤2: 生成情节单元")
            
            plot_units = []
            for i, cluster in enumerate(event_clusters):
                if verbose:
                    print(f"  处理聚类 {i+1}/{len(event_clusters)}")
                
                plot_unit = self._generate_plot_unit(cluster)
                if plot_unit and "error" not in plot_unit:
                    plot_units.append(plot_unit)
                    if verbose:
                        print(f"    ✓ 生成Plot: {plot_unit.get('title', 'Unknown')}")
                else:
                    if verbose:
                        print(f"    ✗ Plot生成失败: {plot_unit.get('error', 'Unknown error')}")
            
            if not plot_units:
                self.logger.warning("未能生成任何Plot单元")
                return False
            
            self.logger.info(f"成功生成 {len(plot_units)} 个Plot单元")
            
            # 3. 图谱写入 (Graph Construction)
            if verbose:
                print("💾 步骤3: 写入图谱")
            
            success_count = 0
            for i, plot_unit in enumerate(plot_units):
                if verbose:
                    print(f"  写入Plot {i+1}/{len(plot_units)}: {plot_unit.get('title', 'Unknown')}")
                
                if self.neo4j_utils.write_plot_to_neo4j(plot_unit):
                    success_count += 1
                    if verbose:
                        print(f"    ✓ 写入成功")
                else:
                    if verbose:
                        print(f"    ✗ 写入失败")
            
            self.logger.info(f"成功写入 {success_count}/{len(plot_units)} 个Plot")
            
            # 4. 输出统计信息
            if verbose:
                print("📊 步骤4: 统计信息")
                stats = self.neo4j_utils.get_plot_statistics()
                print(f"  Plot节点数: {stats.get('plot_count', 0)}")
                print(f"  关联Event数: {stats.get('event_count', 0)}")
                print(f"  涉及Scene数: {stats.get('scene_count', 0)}")
            
            self.logger.info("Event-Plot-Scene图谱构建完成")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"构建Plot图谱失败: {e}")
            return False
    
    def _identify_plot_candidates(self) -> List[List[str]]:
        """
        识别Plot候选 (事件聚类)
        
        Returns:
            List[List[str]]: 事件聚类列表
        """
        # try:
        # 使用GDS连通分量算法进行聚类
        clusters = self.neo4j_utils.identify_event_clusters_by_connectivity(self.causality_threshold)
        # print("[CHECK] clusters: ", clusters)
        # 过滤聚类大小
        filtered_clusters = []
        for cluster in clusters:
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                filtered_clusters.append(cluster)
            else:
                self.logger.debug(f"过滤聚类 (大小: {len(cluster)}): {cluster}")
        
        self.logger.info(f"聚类完成: {len(clusters)} -> {len(filtered_clusters)} (过滤后)")
        return filtered_clusters
            
        # except Exception as e:
        #     self.logger.error(f"事件聚类失败: {e}")
        #     return []
        
    def generate_plot_id(self, event_cluster: List[str]) -> str:
        """
        生成Plot ID
        
        Args:
            event_cluster: 事件聚类
            
        Returns:
            str: 生成的Plot ID
        """
        import hashlib
        
        # 使用事件ID列表的哈希值生成唯一ID
        event_str = "_".join(sorted(event_cluster))
        hash_obj = hashlib.md5(event_str.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        return f"plot_{hash_hex}"
    
    def _generate_plot_unit(self, event_cluster: List[str]) -> Optional[Dict[str, Any]]:
        """
        生成单个Plot单元
        
        Args:
            event_cluster: 事件聚类
            
        Returns:
            Dict: Plot单元数据，失败时返回None
        """
        try:
            # 获取事件详细信息
            event_details = self.neo4j_utils.get_event_details(event_cluster)
            
            # 获取因果关系路径
            causality_paths = self.neo4j_utils.get_causality_paths(event_cluster)
            
            # 生成Plot ID
            plot_id = self.plot_generator.generate_plot_id(event_cluster)
            
            # 调用Plot生成器
            params = {
                "event_cluster": event_cluster,
                "event_details": event_details,
                "causality_paths": causality_paths
            }
            
            plot_unit = self.plot_generator.call(params)
            
            print("[CHECK] params", params)
            
            print("[CHECK] plot_unit", plot_unit)
            
            if plot_unit and "error" not in plot_unit:
                # 确保Plot有正确的ID
                plot_unit["id"] = plot_id
                plot_unit["event_ids"] = event_cluster
                return plot_unit
            else:
                self.logger.error(f"Plot生成失败: {plot_unit}")
                return None
                
        except Exception as e:
            self.logger.error(f"生成Plot单元失败: {e}")
            return None
    
    def get_plot_summary(self) -> Dict[str, Any]:
        """
        获取Plot图谱摘要信息
        
        Returns:
            Dict: 摘要信息
        """
        try:
            stats = self.neo4j_utils.get_plot_statistics()
            
            # 获取Plot详细信息
            plot_details_cypher = """
            MATCH (p:Plot)
            OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
            RETURN p.id as plot_id, 
                   p.title as plot_title,
                   p.summary as plot_summary,
                   count(e) as event_count
            ORDER BY p.title
            """
            
            plot_details_result = self.neo4j_utils.execute_query(plot_details_cypher)
            plot_details = [dict(record) for record in plot_details_result]
            
            return {
                "statistics": stats,
                "plot_details": plot_details,
                "total_plots": len(plot_details)
            }
            
        except Exception as e:
            self.logger.error(f"获取Plot摘要失败: {e}")
            return {"error": str(e)}
    
    def export_plot_graph(self, output_path: str) -> bool:
        """
        导出Plot图谱数据
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 获取完整的Plot图谱数据
            export_cypher = """
            MATCH (p:Plot)
            OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
            OPTIONAL MATCH (s:Scene)-[:SCENE_CONTAINS]->(e)
            RETURN p.id as plot_id,
                   p.title as plot_title,
                   p.summary as plot_summary,
                   p.structure_type as structure_type,
                   p.narrative_roles as narrative_roles,
                   collect(DISTINCT {
                       event_id: e.id,
                       event_name: e.name,
                       event_description: e.description,
                       scene_id: s.id,
                       scene_name: s.name
                   }) as events
            ORDER BY p.title
            """
            
            result = self.neo4j_utils.execute_query(export_cypher)
            plot_data = [dict(record) for record in result]
            
            # 写入文件
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(plot_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Plot图谱数据已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出Plot图谱失败: {e}")
            return False

