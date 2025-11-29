"""
知识图谱编辑器启动脚本
直接连接到真实的 Neo4j 数据库
"""

from kg_editor_app import launch_editor
from core.utils.neo4j_utils import Neo4jUtils
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore

# 初始化您的配置和工具
# 注意：请确保 config 变量已经在您的环境中定义
# 如果没有，您需要先导入和加载配置
# 例如: from your_config_module import config
from core.utils.config import KAGConfig
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.model_providers.openai_llm import OpenAILLM

config = KAGConfig.from_yaml("configs/config_openai.yaml")

graph_store = GraphStore(config)
vector_store = VectorStore(config, "documents")
doc_type = config.knowledge_graph_builder.doc_type
neo4j_utils = Neo4jUtils(graph_store.driver, doc_type)
neo4j_utils.load_embedding_model(config.graph_embedding)

if __name__ == "__main__":
    print("🚀 启动知识图谱编辑器...")
    print("📍 访问地址: http://localhost:7860")
    print()
    
    # 启动编辑器
    # share=True 可以创建公共链接
    # server_port 可以修改端口号
    launch_editor(neo4j_utils, share=False, server_port=7860)
