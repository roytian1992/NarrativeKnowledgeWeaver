"""
事件因果图构建器
负责构建事件因果关系的有向带权图和情节单元图谱
"""

import json
import pickle
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
from tqdm import tqdm
from pathlib import Path
from kag.llm.llm_manager import LLMManager
from kag.utils.neo4j_utils import Neo4jUtils
from kag.models.data import Entity
from kag.builder.graph_analyzer import GraphAnalyzer
from kag.storage.graph_store import GraphStore
from kag.storage.vector_store import VectorStore
from kag.utils.prompt_loader import PromptLoader
from kag.functions.regular_functions.plot_generation import PlotGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from kag.utils.format import correct_json_format
import logging
from collections import defaultdict
import os
from kag.builder.kg_builder import DOC_TYPE_META

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
    
    def __init__(self, config, doc_type="novel", background_path: str = ""):
        """
        初始化事件因果图构建器
        
        Args:
            config: KAG配置对象
        """
        self.config = config
        self.llm_manager = LLMManager(config)
        self.llm = self.llm_manager.get_llm()
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type)
        self.neo4j_utils.load_emebdding_model(config.memory.embedding_model_name)
        self.event_fallback = [] # 可以加入Goal和Action
        
        if doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {doc_type}")
        self.doc_type = doc_type
        self.meta = DOC_TYPE_META[doc_type]

        # 初始化Plot相关组件
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
        self.prompt_loader = PromptLoader(prompt_dir)
        
        self.background_info = ""
        if background_path:
            print("📖加载背景信息")
            self._load_settings(background_path)
        
        if doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel" 
        self.system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": self.background_info})
        self.graph_analyzer = GraphAnalyzer(config, self.llm)
        
        self.plot_generator = PlotGenerator(self.prompt_loader, self.llm)
        
        # Plot构建配置参数（默认值）
        self.causality_threshold = "Medium"
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        self.logger = logging.getLogger(__name__)        
        self.sorted_scenes = []
        self.event_list = []
        self.event2section_map = {}
        self.allowed_rels = []
        self.max_depth = 3
        self.check_weakly_connected_components = True
        self.min_component_size = 10
        self.max_workers = 32
        self.max_iteration = 5
        
        # 因果关系强度到权重的映射
        self.causality_weight_map = {
            "High": 1.0,
            "Medium": 0.6,
            "Low": 0.3
        }
        
        self.logger.info("EventCausalityBuilder初始化完成")
    
    def _load_settings(self, path: str):
        """
        读取 background + abbreviations，并将其合并到 self.abbreviation_info（一段 Markdown 文本）。

        JSON 结构示例（字段均可选）：
        {
            "background": "……",
            "abbreviations": [
                { "abbr": "UEG", "full": "United Earth Government", "zh": "联合政府", "description": "全球统一政府。" },
                { "symbol": "AI", "meaning": "人工智能", "comment": "广泛应用于…" }
            ]
        }
        """
        self.background_info = ""

        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ---------- 1) 背景段落（可选） ----------
        background = data.get("background", "").strip()
        bg_block = f"**背景设定**：{background}\n" if background else ""

        # ---------- 2) 缩写表（键名宽容） ----------
        def fmt(item: dict) -> str:
            """
            将一个缩写项转为 Markdown 列表条目。任何字段都可选，标题字段优先级为：
            abbr > full > 其他字段 > N/A
            """
            if not isinstance(item, dict):
                return ""

            # 标题字段优先级
            abbr = (
                item.get("abbr")
                or item.get("full")
                or next((v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A")
            )

            # 剩下字段去除标题字段
            parts = []
            for k, v in item.items():
                if k in ("abbr", "full"):
                    continue
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

            return f"- **{abbr}**: " + " - ".join(parts) if parts else f"- **{abbr}**"

        abbr_list = data.get("abbreviations", [])
        abbr_block = "\n".join(fmt(item) for item in abbr_list if isinstance(item, dict))

        if background and abbr_block:
            self.background_info = f"{bg_block}\n{abbr_block}"
        else:
            self.background_info = bg_block or abbr_block
        print(f"✅ 成功从{path}加载背景信息")
    
    def build_event_list(self) -> List[Entity]:
        """
        构建排序后的事件列表
        
        Returns:
            排序后的事件列表
        """
        print("🔍 开始构建事件列表...")
        
        # 1. 获取所有场景并排序
        section_entities = self.neo4j_utils.search_entities_by_type(
            entity_type=self.meta["section_label"]
        )
        
        self.sorted_sections = sorted(
            section_entities,
            key=lambda e: int(e.properties.get("order", 99999))
        )
        
        print(f"✅ 找到 {len(self.sorted_sections )} 个section")
        
        # 2. 从场景中提取事件
        event_list = []
        event2section_map = {}
        
        for scene in tqdm(self.sorted_sections, desc="提取场景中的事件"):
            # 优先查找事件
            results = self.neo4j_utils.search_related_entities(
                source_id=scene.id, 
                predicate=self.meta["contains_pred"], 
                entity_types=["Event"], 
                return_relations=False
            )
            
            # 如果场景中没有事件，则用动作或者目标来填充
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=scene.id, 
                    relation_type=self.meta["contains_pred"], 
                    entity_types=self.event_fallback, 
                    return_relations=False
                )
            
            for result in results:
                if result.id not in event2section_map:
                    event2section_map[result.id] = scene.id
                    event_list.append(result)
        
        self.event_list = event_list
        self.event2section_map = event2section_map
        
        print(f"✅ 构建完成，共找到 {len(event_list)} 个事件")
        return event_list
    
    
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
            # print("[CHECK] res: ", res)
            confidence = res.get("confidence", 0.3)
            rows.append({
                "srcId": src_id,
                "dstId": dst_id,
                "weight": weight,
                "confidence": confidence,
                "reason": res["reason"],
                "predicate": "EVENT_CAUSES"
            })
        self.neo4j_utils.write_event_causes(rows)

    
    def check_causality_batch(
        self,
        pairs: List[Tuple[Entity, Entity]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        批量检查事件对的因果关系（多线程版）

        Args:
            pairs: 事件对列表
            max_workers: 最大并发线程数

        Returns:
            事件对ID到因果关系结果的映射
        """
        print(f"🔍 开始并发检查 {len(pairs)} 对事件的因果关系...")
        causality_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _process_pair(pair: Tuple[Entity, Entity]):
            src_event, tgt_event = pair
            pair_key = (src_event.id, tgt_event.id)
            try:
                # 获取事件信息
                info_1 = self.neo4j_utils.get_entity_info(src_event.id, entity_type="事件", contain_properties=True, contain_relations=True)
                info_2 = self.neo4j_utils.get_entity_info(tgt_event.id, entity_type="事件", contain_properties=True, contain_relations=True)
                
                chunks = self.neo4j_utils.get_entity_by_id(src_event.id).source_chunks + self.neo4j_utils.get_entity_by_id(tgt_event.id).source_chunks
                chunks = list(set(chunks))
                documents = self.vector_store.search_by_ids(chunks)
                results = {doc.content for doc in documents}
                related_context = "\n".join(list(results))
                
                # 调用 extractor 检查因果关系
                result_json = self.graph_analyzer.check_event_causality(
                    info_1, info_2, system_prompt=self.system_prompt_text, related_context=related_context
                )
                result_dict = json.loads(result_json)
                # print("[CHECK] result_dict: ", result_dict)
                return pair_key, {
                    'src_event': src_event,
                    'tgt_event': tgt_event,
                    'causal': result_dict.get('causal', 'Low'),
                    'reason': result_dict.get('reason', ''),
                    'reverse': result_dict.get('reverse', False),
                    'confidence': result_dict.get('confidence', 0.3),
                    'raw_result': result_json
                }
                
            except Exception as e:
                # 出错时返回 Low 强度且记录错误
                return pair_key, {
                    'src_event': src_event,
                    'tgt_event': tgt_event,
                    'causal': 'Low',
                    'reason': f'检查过程出错: {e}',
                    'reverse': False,
                    'confidence': result_dict.get('confidence', 0),
                    'raw_result': ''
                }

        # 并发执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(_process_pair, pair): pair for pair in pairs
            }
            for fut in tqdm(as_completed(future_to_pair),
                            total=len(future_to_pair),
                            desc="检查因果关系"):
                key, res = fut.result()
                causality_results[key] = res

        print(f"✅ 因果关系并发检查完成")
        return causality_results
        
    def sort_event_pairs_by_section_order(
        self, pairs: List[Tuple[Entity, Entity]]
    ) -> List[Tuple[Entity, Entity]]:
        def get_order(evt: Entity) -> int:
            sec_id = self.event2section_map.get(evt.id)
            if not sec_id:
                return 99999
            sec = self.neo4j_utils.get_entity_by_id(sec_id)
            return int(sec.properties.get("order", 99999))

        ordered = []
        for e1, e2 in pairs:
            ordered.append((e1, e2) if get_order(e1) <= get_order(e2) else (e2, e1))
        return ordered

    def initialize(self):
        # 1. 创建子图和计算社区划分
        self.neo4j_utils.delete_relation_type("EVENT_CAUSES")
        self.neo4j_utils.create_subgraph(
            graph_name="event_graph",
            exclude_entity_types=[self.meta["section_label"]],
            exclude_relation_types=[self.meta["contains_pred"], "EVENT_CAUSES"],
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
            reachable = self.neo4j_utils.check_nodes_reachable(src_id, tgt_id, excluded_rels=[self.meta["contains_pred"], "EVENT_CAUSES"])
            if reachable: # 如果节点间距离小于3，保留。
                filtered_pairs.append(pair)
            else:
                score = self.neo4j_utils.compute_semantic_similarity(src_id, tgt_id)
                if score >= 0.7: # 如果节点间的相似度大于等于0.7，保留。
                    filtered_pairs.append(pair)  
        return filtered_pairs
    
    def build_event_causality_graph(
        self,
        limit_events: Optional[int] = None
    ) -> None:
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
        filtered_pairs = self.sort_event_pairs_by_section_order(filtered_pairs)
        print("     最终候选事件对数量： ", len(filtered_pairs))
        # 5. 检查因果关系
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)
        
        # 6. 写回 EVENT_CAUSES
        print("\n🔗 写回 EVENT_CAUSES 关系...")
        self.write_event_cause_edges(causality_results)
        self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True)

    def detect_flattened_causal_patterns(self, edges: List[Dict]) -> List[Dict]:
        """
        从边集中发现类似 A→B, A→C, A→D 且存在 B→D 的冗余结构，用于后续因果链精炼

        Returns:
            List of {
                "source": A,
                "targets": [B, C, D],
                "internal_links": [(B, D), (C, D)]
            }
        """
        # 构建邻接表和反向边集合
        forward_graph = defaultdict(set)
        edge_set = set()

        for edge in edges:
            sid = edge["sid"]
            tid = edge["tid"]
            forward_graph[sid].add(tid)
            edge_set.add((sid, tid))

        patterns = []

        for a, a_children in forward_graph.items():
            a_children = list(a_children)
            if len(a_children) < 2:
                continue  # 至少两个指向才可能构成该模式

            internal_links = []
            for i in range(len(a_children)):
                for j in range(len(a_children)):
                    if i == j:
                        continue
                    u, v = a_children[i], a_children[j]
                    if (u, v) in edge_set:
                        internal_links.append((u, v))

            if internal_links:
                patterns.append({
                    "source": a,
                    "targets": a_children,
                    "internal_links": internal_links
                })

        # print(f"[+] Detected {len(patterns)} flattened causal patterns")
        return patterns
    
    def filter_weak_edges_in_patterns(
        self,
        patterns: List[Dict],
        edge_map: Dict[Tuple[str, str], Dict],
        weight_threshold: float = 0.3,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        从 flattened patterns 中剔除 weight 和 confidence 都偏低的边
        """
        cleaned_patterns = []
        # print("[CHECK] patterns: ", patterns)
        for pat in patterns:
            src = pat["source"]
            targets = pat["targets"]
            internals = pat["internal_links"]

            # 过滤 source → target 边
            new_targets = []
            for t in targets:
                info = edge_map.get((src, t))
                confidence = info.get("confidence", 0) or 0
                # print("[CHECK] confidence: ", confidence)
                if not info:
                    continue
                if not (info["weight"] <= weight_threshold and confidence  < conf_threshold):
                    new_targets.append(t)

            # 过滤 internal 边
            new_internals = []
            for u, v in internals:
                info = edge_map.get((u, v))
                confidence = info.get("confidence", 0) or 0
                if not info:
                    continue
                if not (info["weight"] <= weight_threshold and confidence < conf_threshold):
                    new_internals.append((u, v))

            # 保留结构
            if len(new_targets) >= 2 and new_internals:
                cleaned_patterns.append({
                    "source": src,
                    "targets": new_targets,
                    "internal_links": new_internals
                })

        # print(f"[+] Filtered to {len(cleaned_patterns)} refined patterns")
        return cleaned_patterns
    
    def collect_removed_edges(self,
        original_patterns: List[Dict],
        filtered_patterns: List[Dict]
    ) -> Set[Tuple[str, str]]:
        """
        比较两组 pattern 结构，收集被删除导致结构变化的边

        Returns:
            被标记为删除候选的边集合（sid, tid）
        """
        # 抽取原始结构中的全部边
        def extract_edges(patterns: List[Dict]) -> Set[Tuple[str, str]]:
            edge_set = set()
            for pat in patterns:
                src = pat["source"]
                for tgt in pat["targets"]:
                    edge_set.add((src, tgt))
                edge_set.update(pat["internal_links"])
            return edge_set

        origin_edges = extract_edges(original_patterns)
        filtered_edges = extract_edges(filtered_patterns)

        removed_edges = origin_edges - filtered_edges
        print(f"[+] Found {len(removed_edges)} candidate edges removed due to pattern collapse")
        return list(removed_edges)
    
    def filter_pattern(self, pattern, edge_map):
        source = pattern["source"]
        targets = pattern["targets"]
        internal_links = pattern["internal_links"]
        context_to_check = []
        for link in internal_links:
            mid_tgt_sim = self.neo4j_utils.compute_semantic_similarity(link[0], link[1])
            src_mid_sim = self.neo4j_utils.compute_semantic_similarity(source, link[0])
            src_tgt_sim = self.neo4j_utils.compute_semantic_similarity(source, link[1])
            
            mid_tgt_conf = edge_map.get((link[0], link[1]))["confidence"]
            src_mid_conf = edge_map.get((source, link[0]))["confidence"]
            src_tgt_conf = edge_map.get((source, link[1]))["confidence"]
            
            # print(source_mid_score, internal_score, source_target_score)
            if (src_mid_sim > src_tgt_sim and mid_tgt_sim > src_tgt_sim) or (src_mid_conf > src_tgt_conf and mid_tgt_conf > src_tgt_conf) :
                context_to_check.append({
                    "entities": [source, link[0], link[1]],
                    "details": [
                        {"edge": [source, link[0]], "similarity": src_mid_sim, "confidence": src_mid_conf},
                        {"edge": [source, link[1]], "similarity": src_tgt_sim, "confidence": src_tgt_conf},
                        {"edge": [link[0], link[1]], "similarity": mid_tgt_sim, "confidence": mid_tgt_conf},
                    ]
                })
                
        return context_to_check
    
    
    def prepare_context(self, pattern_detail):
        event_details = self.neo4j_utils.get_event_details(pattern_detail["entities"])
        full_event_details = "三个事件实体的描述如下：\n"
        for i, event_info in enumerate(event_details):
            event_id = event_info["event_id"]
            full_event_details += f"**事件{i+1}的相关描述如下：**\n事件id：{event_id}\n"
            
            background = self.neo4j_utils.get_entity_info(event_id, "事件", True, True)
            event_props = json.loads(event_info.get("event_properties"))
            # print(event_props)
            non_empty_props = {k: v for k, v in event_props.items() if isinstance(v, str) and v.strip()}

            if non_empty_props:
                background += "\n事件的属性如下：\n"
                for k, v in non_empty_props.items():
                    background += f"- {k}：{v}\n"

            if i+1 !=  len(event_details):
                background += "\n"
            full_event_details += background
        
        full_relation_details = "它们之间已经存在的因果关系有：\n"
        relation_details = pattern_detail["details"]
        for i, relation_info in enumerate(relation_details):
            src, tgt = relation_info["edge"]
            background = f"{i+1}. " + self.neo4j_utils.get_relation_summary(src, tgt, "EVENT_CAUSES")
            background += f"\n关系的语义相似度为：{round(relation_info["similarity"], 4)}，置信度为：{relation_info["confidence"]}。"
            if i+1 !=  len(relation_details):
                background += "\n\n"
            full_relation_details += background
        return full_event_details, full_relation_details
    

    def run_SABER(self):
        """
        执行基于结构+LLM的因果边精简优化过程
        """
        loop_count = 0
        while True:
            print(f"\n===== [第 {loop_count + 1} 轮优化] =====")

            # === 获取连通体（优先 SCC，再选 WCC） ===
            scc_components = self.neo4j_utils.fetch_scc_components("event_causality_graph", 2)
            wcc_components = []
            if self.check_weakly_connected_components:
                wcc_components = self.neo4j_utils.fetch_wcc_components("event_causality_graph", self.min_component_size)

            connected_components = scc_components + wcc_components
            print(f"📌 当前连通体数量：SCC={len(scc_components)}，WCC={len(wcc_components)}")

            # === 构造所有 triangle 和边信息 ===
            all_triangles = []
            edge_map_global = {}

            for cc in connected_components:
                node_map, edges = self.neo4j_utils.load_connected_components_subgraph(cc)
                edge_map = {
                    (e["sid"], e["tid"]): {"weight": e["weight"], "confidence": e.get("confidence", 1.0)}
                    for e in edges
                }
                edge_map_global.update(edge_map)

                old_patterns = self.detect_flattened_causal_patterns(edges)
                new_patterns = self.filter_weak_edges_in_patterns(old_patterns, edge_map)
                for pattern in new_patterns:
                    all_triangles += self.filter_pattern(pattern, edge_map)

            print(f"🔺 本轮需判断的三元因果结构数量：{len(all_triangles)}")

            # === ✅ 提前退出条件 ===
            if loop_count >= 1:
                if len(scc_components) == 0 and len(set(removed_edges)) == 0:
                    print("✅ 图结构已无强连通体，且无待判定三元结构，任务终止。")
                    break
            elif loop_count >= self.max_iteration:
                break

            # === 并发处理三元结构 ===
            removed_edges = []

            def process_triangle(triangle_):
                try:
                    event_details, relation_details = self.prepare_context(triangle_)
                    chunks = [self.neo4j_utils.get_entity_by_id(ent_id).source_chunks[0] for ent_id in triangle_["entities"]]
                    chunks = list(set(chunks))
                    documents = self.vector_store.search_by_ids(chunks)
                    results = {doc.content for doc in documents}
                    related_context = "\n".join(list(results))
                    # related_context = "" # 为了速度

                    output = self.graph_analyzer.evaluate_event_redundancy(
                        event_details, relation_details, self.system_prompt_text, related_context
                    )
                    output = json.loads(correct_json_format(output))
                    if output.get("remove_edge", False):
                        return (triangle_["entities"][0], triangle_["entities"][2])
                except Exception as e:
                    print(f"[⚠️ 错误] Triangle 判断失败: {triangle_['entities']}, 错误信息: {str(e)}")
                return None

            print(f"🧠 正在并发判断三元结构...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_triangle, tri) for tri in all_triangles]
                for f in tqdm(as_completed(futures), total=len(futures), desc="LLM判断"):
                    res = f.result()
                    if res:
                        removed_edges.append(res)

            print(f"❌ 本轮待定移除边数量：{len(set(removed_edges))}")

            # === 删除边 ===
            for edge in removed_edges:
                self.neo4j_utils.delete_relation_by_ids(edge[0], edge[1], "EVENT_CAUSES")

            # === 刷新 GDS 图 ===
            # self.neo4j_utils.create_event_causality_graph("event_causality_graph", min_confidence=0.5, min_weight=0.5, force_refresh=True)
            self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True)
            loop_count += 1

    def get_all_event_chains(self, min_weight: float = 0.0, min_confidence: float = 0.0):
        """
        获取所有可能的事件链（从起点到没有出边的终点）
        """
        starting_events = self.neo4j_utils.get_starting_events()
        chains = []
        for event in starting_events:
            all_chains = self.neo4j_utils.find_event_chain(event, min_weight, min_confidence)
            chains.extend([chain for chain in all_chains if len(chain) >= 1])
        return chains
    
    def prepare_chain_context(self, chain):
        if len(chain) > 1:
            context = "事件链：" + "->".join(chain) +"\n\n事件具体信息如下：\n"
        else:
            context = f"事件：{chain[0]}" +"\n\n事件具体信息如下：\n"
        for i, event in enumerate(chain):
            context += f"事件{i+1}：{event}\n" + self.neo4j_utils.get_entity_info(event, "事件", False, True) + "\n"
        return context
        

    def generate_plot_relations(self):
        
        self.neo4j_utils.process_all_embeddings(entity_types=["Plot"])
        
        all_plot_pairs = self.neo4j_utils.get_plot_pairs()
        edges_to_add = []

        def process_pair(pair):
            try:
                plot_A_info = self.neo4j_utils.get_entity_info(pair["src"], "情节", False, True)
                plot_B_info = self.neo4j_utils.get_entity_info(pair["tgt"], "情节", False, True)
                result = self.graph_analyzer.extract_plot_relation(plot_A_info, plot_B_info, self.system_prompt_text)
                result = json.loads(correct_json_format(result))

                pair_edges = []
                if result["relation_type"] == "PLOT_CONTRIBUTES_TO":
                    first = result["direction"].split("->")[0]
                    if first == "A":
                        pair_edges.append({
                            "src": pair["src"],
                            "tgt": pair["tgt"],
                            "relation_type": result["relation_type"],
                            "confidence": result["confidence"],
                            "reason": result["reason"]
                        })
                    else:
                        pair_edges.append({
                            "src": pair["tgt"],
                            "tgt": pair["src"],
                            "relation_type": result["relation_type"],
                            "confidence": result["confidence"],
                            "reason": result["reason"]
                        })
                elif result["relation_type"] == "PLOT_CONFLICTS_WITH":
                    pair_edges.append({
                        "src": pair["src"],
                        "tgt": pair["tgt"],
                        "relation_type": result["relation_type"],
                        "confidence": result["confidence"],
                        "reason": result["reason"]
                    })
                    pair_edges.append({
                        "src": pair["tgt"],
                        "tgt": pair["src"],
                        "relation_type": result["relation_type"],
                        "confidence": result["confidence"],
                        "reason": result["reason"]
                    })
                return pair_edges
            except Exception as e:
                print(f"[⚠] 处理情节对 {pair} 出错: {e}")
                return []

        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_pair, pair) for pair in all_plot_pairs]
            # for future in as_completed(futures):
            for future in tqdm(as_completed(futures), total=len(futures), desc="抽取情节关系"):
                edges_to_add.extend(future.result())

        # 批量写入 Neo4j
        if edges_to_add:
            self.neo4j_utils.create_plot_relations(edges_to_add)
            print(f"[✓] 已创建情节关系 {len(edges_to_add)} 条")
        else:
            print("[!] 没有生成任何情节关系")

    
    def build_event_plot_graph(self):
        all_chains = self.get_all_event_chains(0.5, 0.5)
        print("[✓] 当前事件链数量：", len(all_chains))
        self.neo4j_utils.reset_event_plot_graph()
        def process_chain(chain):
            try:
                event_chain_info = self.prepare_chain_context(chain)
                chunks = [self.neo4j_utils.get_entity_by_id(ent_id).source_chunks[0] for ent_id in chain]
                chunks = list(set(chunks))
                documents = self.vector_store.search_by_ids(chunks)
                results = {doc.content for doc in documents}
                related_context = "\n".join(list(results))

                result = self.graph_analyzer.generate_event_plot(
                    event_chain_info=event_chain_info,
                    system_prompt=self.system_prompt_text,
                    related_context=related_context
                )
                result = json.loads(correct_json_format(result))
                if result["is_plot"]:
                    plot_info = result["plot_info"]
                    plot_title = plot_info["title"]
                    plot_info["id"] = f"plot_{hash(f'{plot_title}') % 1_000_000}"
                    plot_info["event_ids"] = chain
                    plot_info["reason"] = result.get("reason", "")
                    self.neo4j_utils.write_plot_to_neo4j(plot_data=plot_info)
                    return True
                return False
            except Exception as e:
                print(f"[!] 处理事件链 {chain} 时出错: {e}")
                return False

        success_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_chain, chain) for chain in all_chains]
            for future in tqdm(as_completed(futures), total=len(futures), desc="并发生成情节图谱"):
                if future.result():
                    success_count += 1

        print(f"[✓] 成功生成情节数量：{success_count}/{len(all_chains)}")

                
                
        
        