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
        max_depth: int = 3
    ) -> List[Tuple[Entity, Entity]]:
        """
        利用 Neo4j 中 Louvain 结果直接筛选同社区且 max_depth 内可达的事件对
        """
        # 把事件 ID 做成集合，便于后面实体映射
        id2entity = {e.id: e for e in events}

        pairs = self.neo4j_utils.fetch_event_pairs_same_community(
            max_depth=max_depth
        )
        # print("[CHECK]: ", pairs)
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))

        print(f"[✓] 同社区 + 可达事件对: {len(filtered_pairs)}")
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
        self.neo4j_utils.create_subgraph(
            graph_name="event_graph",
            exclude_node_labels=["Scene"],
            exclude_rel_types=["SCENE_CONTAINS"],
            force_refresh=True
        )

        self.neo4j_utils.run_louvain(
            graph_name="event_graph",
            write_property="community",
            force_run=True
        )
    
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
        filtered_pairs = self.sort_event_pairs_by_scene_time(filtered_pairs)
        
        # 5. 检查因果关系
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)
        
        # 6. 写回 EVENT_CAUSES
        print("\n🔗 写回 EVENT_CAUSES 关系...")
        self.write_event_cause_edges(causality_results)
    
