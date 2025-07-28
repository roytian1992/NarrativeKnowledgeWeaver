"""
事件因果图构建器
负责构建事件因果关系的有向带权图
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

class EventCausalityBuilder:
    """
    事件因果图构建器
    
    主要功能：
    1. 从Neo4j加载和排序事件
    2. 通过连通体和社区过滤事件对
    3. 使用extractor检查因果关系
    4. 构建有向带权NetworkX图
    5. 保存和加载图数据
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
            entity_type="Scene", 
            limit=500
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
                predicate="SCENE_CONTAINS", 
                entity_types=["Event"], 
                return_relations=False
            )
            
            # 如果场景中没有事件，则用动作或者目标来填充
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=scene.id, 
                    predicate="SCENE_CONTAINS", 
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
    
    def get_event_info(self, event_id: str, event_tag: int = 1) -> str:
        """
        获取事件的详细信息，用于因果关系检查
        
        Args:
            event_id: 事件ID
            event_tag: 事件标签（用于区分事件1和事件2）
            
        Returns:
            格式化的事件信息字符串
        """
        event_node = self.neo4j_utils.get_entity_by_id(event_id)
        if not event_node:
            return f"事件{event_tag}：未找到事件信息"
        
        results = self.neo4j_utils.search_related_entities(
            source_id=event_id, 
            return_relations=True
        )
        
        relevant_info = []
        for result in results:
            info = self._get_relation_info(result[1])
            if info:
                relevant_info.append(info)
        
        context = (
            f"事件{event_tag}（{event_node.name}）：{event_node.properties.get('description', '无具体描述')}\n"
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
    ) -> List[Tuple[Entity, Entity]]:
        """
        通过连通体和社区过滤事件对
        
        Args:
            events: 事件列表
            
        Returns:
            过滤后的事件对列表
        """
        print("🔍 开始社区过滤...")
        
        # 1. 获取允许的关系类型（排除SCENE_CONTAINS）
        self.allowed_rels = self.neo4j_utils.list_relationship_types()
        if "SCENE_CONTAINS" in self.allowed_rels:
            self.allowed_rels.remove("SCENE_CONTAINS")
        
        print(f"✅ 使用关系类型: {len(self.allowed_rels)} 种")
        
        # 2. 构建过滤后的图
        G = self.neo4j_utils.build_filtered_graph(set(self.allowed_rels))
        print(f"✅ 图构建完成，节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")
        
        # 3. 执行连通体 + Louvain 社区划分
        print("🔍 执行连通体 + Louvain 社区划分中...")
        node_cluster_map = self.neo4j_utils.assign_components_and_communities(G)
        
        print(f"✅ 划分完成，共有 {len(set(c for c, _ in node_cluster_map.values()))} 个连通体，"
              f"{len(set((c, comm) for c, comm in node_cluster_map.values()))} 个社区")
        
        # 4. 筛选同一社区内的事件对
        exist_count = 0
        nonexist_count = 0
        total_pairs = 0
        accepted_pairs = []
        
        print("🔍 开始筛选社区内部事件对...")
        
        for i in tqdm(range(len(events))):
            for j in range(i + 1, len(events)):
                e1 = events[i]
                e2 = events[j]

                key1 = node_cluster_map.get(e1.id)
                key2 = node_cluster_map.get(e2.id)

                if not key1 or not key2:
                    continue

                if key1 == key2:
                    exist_count += 1
                    accepted_pairs.append((e1, e2))
                else:
                    nonexist_count += 1

                total_pairs += 1
        
        print(f"✅ 总共对比: {total_pairs} 对")
        print(f"✅ 同一社区内: {exist_count} 对")
        print(f"❌ 不同社区跳过: {nonexist_count} 对")
        
        # 5. 进一步通过路径连通性过滤
        filtered_pairs = []
        
        for src_node, tgt_node in tqdm(accepted_pairs, desc="🔍 筛选路径连通的事件对"):
            src_id, tgt_id = src_node.id, tgt_node.id

            # 连通体很小，直接通过
            component_id, _ = node_cluster_map[src_id]
            component_size = sum(1 for c, _ in node_cluster_map.values() if c == component_id)
            if component_size < self.max_depth:
                filtered_pairs.append((src_node, tgt_node))
                continue

            # 进一步在 max_depth 内搜索路径
            if self.neo4j_utils.has_path_between_nx(G, src_id, tgt_id, max_depth=self.max_depth):
                filtered_pairs.append((src_node, tgt_node))
        
        print(f"✅ 路径过滤完成，最终保留 {len(filtered_pairs)} 对事件")
        return filtered_pairs
    
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
            event_1_info = self.get_event_info(src_event.id, 1)
            event_2_info = self.get_event_info(tgt_event.id, 2)
            
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
    
    def build_causality_graph(
        self, 
        causality_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        构建事件因果关系的有向带权图
        
        Args:
            causality_results: 因果关系检查结果
            
        Returns:
            NetworkX有向图
        """
        print("🔍 开始构建因果关系图...")
        
        G = nx.DiGraph()
        
        # 添加节点和边
        for pair_key, result in causality_results.items():
            src_id, tgt_id = pair_key
            src_event = result['src_event']
            tgt_event = result['tgt_event']
            causal_level = result['causal']
            reason = result['reason']
            reverse = result.get('reverse', False)  # 获取reverse字段
            
            # 根据scene_id排序决定初始边的方向
            src_scene_id = self.event2scene_map.get(src_id)
            tgt_scene_id = self.event2scene_map.get(tgt_id)
            
            if src_scene_id and tgt_scene_id:
                src_scene = self.neo4j_utils.get_entity_by_id(src_scene_id)
                tgt_scene = self.neo4j_utils.get_entity_by_id(tgt_scene_id)
                
                if src_scene and tgt_scene:
                    # 比较scene_id和sub_scene_id
                    src_scene_num = int(src_scene.properties.get("scene_number", 0))
                    src_sub_scene_num = int(src_scene.properties.get("sub_scene_number", 0))
                    tgt_scene_num = int(tgt_scene.properties.get("scene_number", 0))
                    tgt_sub_scene_num = int(tgt_scene.properties.get("sub_scene_number", 0))
                    
                    # 确定初始边的方向：较早的事件指向较晚的事件
                    if (src_scene_num, src_sub_scene_num) <= (tgt_scene_num, tgt_sub_scene_num):
                        from_id, to_id = src_id, tgt_id
                        from_event, to_event = src_event, tgt_event
                        from_scene, to_scene = src_scene, tgt_scene   
                    else:
                        from_id, to_id = tgt_id, src_id
                        from_event, to_event = tgt_event, src_event
                        from_scene, to_scene = tgt_scene, src_scene 
                    
                    # 根据reverse字段决定是否反转方向
                    if reverse:
                        # 如果reverse为True，反转因果方向
                        from_id, to_id = to_id, from_id
                        from_event, to_event = to_event, from_event
                        from_scene, to_scene = to_scene, from_scene
                        print(f"🔄 反转因果方向: {from_event.name} -> {to_event.name}")
                    
                    # 添加节点（如果不存在）
                    if not G.has_node(from_id):
                        G.add_node(from_id, 
                                  name=from_event.name,
                                  description=from_event.properties.get("description", ""),
                                  # scene_id=self.event2scene_map.get(from_id),
                                  scene_name = from_scene.name if from_scene else None,
                                  entity_type=from_event.type)
                    
                    if not G.has_node(to_id):
                        G.add_node(to_id, 
                                  name=to_event.name,
                                  description=to_event.properties.get("description", ""),
                                  # scene_id=self.event2scene_map.get(to_id),
                                  scene_name = to_scene.name if to_scene else None,
                                  entity_type=to_event.type)
                    
                    # 添加边（带权重）
                    weight = self.causality_weight_map.get(causal_level, 0.3)
                    G.add_edge(from_id, to_id,
                              weight=weight,
                              causal_level=causal_level,
                              reason=reason,
                              reverse=reverse,  # 保存reverse信息
                              raw_result=result.get('raw_result', ''))
        
        print(f"✅ 因果关系图构建完成")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")
        
        # 统计因果关系强度分布
        causal_levels = [data['causal_level'] for _, _, data in G.edges(data=True)]
        level_counts = {level: causal_levels.count(level) for level in set(causal_levels)}
        print(f"   因果关系强度分布: {level_counts}")
        
        # 统计反转边的数量
        reversed_edges = [data for _, _, data in G.edges(data=True) if data.get('reverse', False)]
        print(f"   反转边数量: {len(reversed_edges)}")
        
        return G
    
    def save_graph(self, graph: nx.DiGraph, filepath: str):
        """
        保存图到文件
        
        Args:
            graph: NetworkX有向图
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"✅ 图已保存到: {filepath}")
    
    def load_graph(self, filepath: str, format: str = 'graphml') -> nx.DiGraph:
        """
        从文件加载图
        
        Args:
            filepath: 文件路径
            
        Returns:
            NetworkX有向图
        """
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
            
        print(f"✅ 图已从 {filepath} 加载")
        return graph
    
    def build_complete_causality_graph(
        self,
        limit_events: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        完整的事件因果图构建流程
        
        Args:
            limit_events: 限制处理的事件数量（用于测试）
            
        Returns:
            构建完成的NetworkX有向图
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
        filtered_pairs = self.filter_event_pairs_by_community(
            event_list
        )
        
        if not filtered_pairs:
            print("❌ 没有找到符合条件的事件对，返回空图")
            return nx.DiGraph()
        
        # 5. 检查因果关系
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)
        
        # 6. 构建因果图
        print("\n🔍 构建因果图...")
        causality_graph = self.build_causality_graph(causality_results)
        
        # 7. 保存图（如果指定了输出路径）
        output_path = "data/event_causality_graph/event_causality_graph.pickle"
        print(f"\n💾 保存图到 {output_path}...")
        self.save_graph(causality_graph, output_path)
        
        # 8. 输出统计信息
        print("\n📊 图统计信息:")
        stats = self.get_graph_statistics(causality_graph)
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ 事件因果图构建完成！")
        return causality_graph

    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        获取图的统计信息
        
        Args:
            graph: NetworkX有向图
            
        Returns:
            统计信息字典
        """
        stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph),
            'strongly_connected_components': nx.number_strongly_connected_components(graph),
            'weakly_connected_components': nx.number_weakly_connected_components(graph)
        }
        
        # 因果关系强度分布
        if graph.number_of_edges() > 0:
            causal_levels = [data['causal_level'] for _, _, data in graph.edges(data=True)]
            stats['causal_level_distribution'] = {
                level: causal_levels.count(level) for level in set(causal_levels)
            }
            
            # 反转边统计
            reversed_edges = [data for _, _, data in graph.edges(data=True) if data.get('reverse', False)]
            stats['reversed_edges_count'] = len(reversed_edges)
            stats['reversed_edges_percentage'] = len(reversed_edges) / graph.number_of_edges() * 100
        
        return stats

