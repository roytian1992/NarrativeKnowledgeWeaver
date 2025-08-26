from core.agent.retriever_agent import QuestionAnsweringAgent
from core.utils.config import KAGConfig
import os
import json
from core.builder.graph_builder import KnowledgeGraphBuilder
from core.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document


config = KAGConfig.from_yaml("configs/config_openai.yaml")
builder = KnowledgeGraphBuilder(config, doc_type="screenplay")
builder.initialize_agents()

# 加载数据
base = config.storage.knowledge_graph_path
all_documents = [TextChunk(**o) for o in
            json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

# 测试抽取
results = builder.information_extraction_agent.run(all_documents[0].content)
# results["entities"] 和 results["relations"] 获取结果


builder.information_extraction_agent.set_mode("probing") # 设置探索模式，即不思考、不反思重试，非智能体模式（可能会用到初始的洞见）
# results = builder.information_extraction_agent.run(all_documents[0].content)

# 生成schema
# 其它参数： schema: Dict = {}, background: str = "", abbreviations: List = [] 也可以提供当前的
schema, settings = builder.update_schema(sample_ratio=0.35, documents=all_documents) # 不能jupyter 里面运行

# 问答
agent = QuestionAnsweringAgent(config, doc_type="screenplay") 
responses = agent.ask("马卡洛夫出现在了哪些场次和场景里？")

final_text = agent.extract_final_text(responses) # 最终的答案
# print(final_text)
tool_uses = agent.extract_tool_uses(responses) # 中间使用的工具 [{"tool_name": tool_name, "tool_arguments": '{tool params}', "tool_output": tool_output}, ...]

