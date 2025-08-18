#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAG-Builder 主程序
命令行工具, 用于从JSON文件构建知识图谱
"""

import argparse
import sys
from pathlib import Path
from core.builder.narrative_graph_builder import EventCausalityBuilder

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import KAGConfig
from core.builder.graph_builder import KnowledgeGraphBuilder
import os
import logging

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)

# os.environ["HTTP_PROXY"] = "http://localhost:7890"
# os.environ["HTTPS_PROXY"] = "http://localhost:7890"

def main():
    parser = argparse.ArgumentParser(description="KAG-Builder: 知识图谱构建器")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    parser.add_argument(
        "--output-stats", "-s",
        help="输出统计信息到文件"
    )
    
    args = parser.parse_args()
    
    # try:
    # 加载配置
    config = KAGConfig.from_yaml(args.config)
    print(f"🔧 使用配置文件: {config}")
    if args.verbose:
        print(f"✅ 从 {args.config} 加载配置")

    
    # 创建构建器
    builder = KnowledgeGraphBuilder(config)
    
    # 构建知识图谱
    # builder.prepare_chunks(args.input, verbose=args.verbose) # 文本准备阶段：长文本拆分和元数据标注
    # builder.store_chunks(verbose=args.verbose) # 保存近向量数据库和本地json
    # builder.run_graph_probing(verbose=args.verbose, sample_ratio=0.35) # from_scratch 的时候建议 0.35
    builder.initialize_agents()
    # builder.extract_entity_and_relation(verbose=args.verbose) # 实体和关系抽取
    builder.run_extraction_refinement(verbose=args.verbose)
    builder.extract_entity_attributes(verbose=args.verbose) # 实体消歧和属性抽取
    kg = builder.build_graph_from_results(verbose=args.verbose)
    builder.prepare_graph_embeddings()

    # 可以插入一个定制化的小图谱来补充一些东西
    # additional_builder = ...
    
    
    event_graph_builder = EventCausalityBuilder(config)
    # event_graph_builder.initialize() # 初始化事件情节图
    # event_graph_builder.build_event_causality_graph() # 开始构建事件因果图
    # event_graph_builder.run_SABER() # 运行断环、去冗余算法
    event_graph_builder.build_event_plot_graph() # 构建情节-事件图
    event_graph_builder.generate_plot_relations() # 抽取情节间关系
    event_graph_builder.prepare_graph_embeddings()
    
    # # 输出统计信息
    # stats = builder.get_stats()
    # print("\n📊 最终统计信息:")
    # print(f"   - 实体数量: {stats['knowledge_graph']['entities']}")
    # print(f"   - 关系数量: {stats['knowledge_graph']['relations']}")
    # print(f"   - 文档数量: {stats['knowledge_graph']['documents']}")
    # print(f"   - 文本块数量: {stats['knowledge_graph']['chunks']}")
    
    # # 保存统计信息
    # if args.output_stats:
    #     import json
    #     with open(args.output_stats, 'w', encoding='utf-8') as f:
    #         json.dump(stats, f, ensure_ascii=False, indent=2)
    #     print(f"📄 统计信息已保存到: {args.output_stats}")
    
    print("\n🎉 知识图谱和情节事件图构建完成!")
        
    # except Exception as e:
    #     print(f"❌ 构建失败: {str(e)}")
    #     sys.exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # ✅ 解决CUDA多进程问题
    main()


