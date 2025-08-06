# KAG 智能知识图谱构建与事件因果分析系统说明

## 项目概述

[`kag`](./kag) 目录是 **KGAG (Knowledge Graph and Agent Generation)** 项目的核心实现，提供了从非结构化文本自动构建知识图谱并生成事件因果图谱的完整流程。  
系统结合 **大语言模型（LLM）**、**LangChain / LangGraph** 框架与 **Neo4j 图数据库**，可对小说、剧本等文档进行 **实体（Entity）**、**关系（Relation）** 和 **属性（Attribute）** 抽取，构建结构化的知识图谱。  
在此基础上，系统可以分析事件的先后与因果逻辑，生成 **事件因果链（Event Causality Graph）** 和 **情节单元图谱（Plot Graph）**。

与仓库根目录的原始 `README.md` 相比，本说明专注于 `kag` 代码实现，帮助开发者快速理解模块功能与调用方式。

---

## 目录结构

```
kag/
│
├── agent/                  # 基于 LangGraph / LLM 的智能 Agent 封装
│   ├── attribute_extraction_agent.py   # 属性抽取 Agent
│   ├── kg_extraction_agent.py          # 实体+关系抽取 Agent
│   ├── openai_agent.py                 # OpenAI API Agent 封装
│   └── qwen3_fncall_agent.py            # Qwen 本地 Function Calling Agent
│
├── builder/                # 图谱构建与分析模块
│   ├── kg_builder.py                    # 知识图谱构建流程（实体、关系）
│   ├── narrative_graph_builder.py       # 叙事结构图谱构建（事件因果、情节单元）
│   └── graph_preprocessor.py            # 图谱预处理（实体消歧、聚类）
│
├── functions/              # LLM 调用的工具函数
│   └── regular_functions/              # 实体、关系、属性等抽取函数
│
├── llm/                     # 大语言模型管理与封装
│   ├── llm_manager.py
│   └── qwen3_llm.py
│
├── storage/                 # 持久化与向量存储
│   ├── vector_memory.py
│   └── graph_store.py
│
└── utils/                   # 通用工具（Neo4j 操作、Prompt 加载、JSON 修复等）
```

---

## 核心功能

1. **文档处理与切分**  
   - 从 JSON 或原始文本加载文档（小说 / 剧本）
   - 并发切分为 chunk（描述 / 对话片段）
   - 支持元数据解析（章节、场景等）

2. **实体、关系、属性抽取**  
   - 基于 LLM（OpenAI 或本地 Qwen）抽取结构化信息  
   - 支持 Function Calling 结构化输出
   - 抽取后进行实体消歧与合并

3. **知识图谱构建**  
   - 将抽取结果写入 Neo4j
   - 为实体与关系生成向量嵌入并创建向量索引
   - 支持跨文档整合

4. **事件因果分析**  
   - 从事件节点构建因果图
   - 过滤无效边与循环（SCC 检测、断环）
   - 聚合情节单元（Plot）

5. **反思与重抽**  
   - 对抽取结果进行质量评分
   - 分数过低时触发重抽，提升精度

---

## 依赖环境

- Python 3.10+
- Neo4j 5.x
- [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph)
- Transformers / vLLM
- SentenceTransformers
- ChromaDB

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 配置

所有运行参数存放于 `config/` 下的 YAML 文件，包括：
- LLM 类型与 API 配置
- Neo4j 连接参数
- 文档类型（`novel` / `screenplay`）
- 并发线程数、显卡分配等

---

## 使用方法

`main.py` 是统一入口，支持不同构建模式：

```bash
# 构建完整知识图谱
python main.py build_kg --config config/kg_config.yaml --json_file data/novel.json

# 构建事件因果图
python main.py build_event_causality --config config/event_config.yaml

# 执行情节单元抽取
python main.py build_plot_graph --config config/plot_config.yaml
```

运行流程示例（`build_kg` 模式）：
1. 加载配置与 LLM
2. 文档切分（并发）
3. 实体、关系抽取（Function Calling）
4. 实体消歧（GraphPreprocessor）
5. 写入 Neo4j 并生成向量索引

---

## 主要类与调用关系

- `KnowledgeGraphBuilder`  
  → `InformationExtractor` 调用实体/关系/属性抽取函数  
  → `Neo4jUtils` 写入图数据库  
  → `VectorMemory` 保存语义向量

- `NarrativeGraphBuilder`  
  → `EventCausalityBuilder` 构建因果图  
  → `PlotGenerator` 聚合情节单元

- `LLMManager`  
  → 管理 OpenAI / Qwen 本地模型加载  
  → 提供统一 `_chat_no_stream` 接口

---

## 输出

- **Neo4j 图谱**（实体、关系、事件因果图）
- **向量索引**（支持语义检索）
- **日志与 JSON 结果文件**

---

## 开发建议

- 对长文档建议开启多进程并分配 GPU，提升并发推理性能
- 对低置信度抽取结果，可通过 `DynamicReflector` 进行二次抽取
- 配合 Neo4j Bloom 或 GraphXR 可视化探索结果


## 安装neo4j
1. 更新系统
sudo apt update && sudo apt upgrade -y

2. 安装依赖
sudo apt install wget apt-transport-https gnupg lsb-release -y

3. 添加 Neo4j 官方 GPG key
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg

4. 添加 Neo4j 源（以 Neo4j 5.x 为例）
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list

5. 安装 Neo4j
sudo apt update
sudo apt install neo4j -y

6. 启动服务并设置开机自启
sudo systemctl enable neo4j
sudo systemctl start neo4j

7. 查看运行状态
sudo systemctl status neo4j

8. 安装gds
cp neo4j-graph-data-science-2.13.4.jar /var/lib/neo4j/plugins/
