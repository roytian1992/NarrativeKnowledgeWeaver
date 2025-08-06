# KAG 智能知识图谱构建与事件因果分析系统说明

## 项目概述

[`kag`](./kag) 目录是 **KGAG (Knowledge Graph and Agent Generation)** 项目的核心实现，提供了从非结构化文本自动构建知识图谱并生成事件因果图谱的完整流程。
系统结合 **大语言模型（LLM）**、**LangChain / LangGraph** 框架与 **Neo4j 图数据库**，可对小说、剧本等文档进行 **实体（Entity）**、**关系（Relation）** 和 **属性（Attribute）** 抽取，构建结构化的知识图谱。
在此基础上，系统可以分析事件的先后与因果逻辑，生成 **事件因果链（Event Causality Graph）** 和 **情节单元图谱（Plot Graph）**。

与仓库根目录的原始 `README.md` 相比，本说明专注于 `kag` 代码实现，帮助开发者快速理解模块功能与调用方式。

---

## 运行示例
先参考configs/config_example.yaml的信息，构建自己的config文件。

```bash
python3 main.py \
    -c configs/config_openai.yaml \
    -i examples/documents/流浪地球2剧本.json \
    -b examples/settings/we2_settings.json \
    -t "screenplay" \
    -v

python3 main.py \
    -c configs/config_openai.yaml \
    -i examples/documents/我机器人.json \
    -b examples/settings/irobot_settings.json \
    -t "novel" \
    -v
```

---

## 命令行参数说明

| 参数                     | 类型   | 说明                                  |
| ---------------------- | ---- | ----------------------------------- |
| `-c, --config`         | str  | 主配置文件路径（YAML 格式），包含 LLM、Neo4j、并发等设置 |
| `-i, --input`          | str  | 输入文档路径（JSON 格式）                     |
| `-b, --build-settings` | str  | 构建流程设置文件路径（YAML/JSON 格式）            |
| `-t, --doc-type`       | str  | 文档类型，可选：`novel` / `screenplay`      |
| `-v, --verbose`        | flag | 是否输出详细日志                            |

---

## 目录结构与功能说明

```
kag/
│
├── agent/                              # 智能 Agent 封装
│   ├── attribute_extraction_agent.py   # 属性抽取 Agent（多轮反思、补充上下文）
│   ├── kg_extraction_agent.py          # 实体+关系抽取 Agent（支持 Function Calling）
│   ├── openai_agent.py                 # OpenAI API Agent 封装
│   └── qwen3_fncall_agent.py            # 本地 Qwen 模型 Function Calling Agent
│
├── builder/
│   ├── kg_builder.py                    # 主知识图谱构建流程（实体、关系、属性）
│   ├── narrative_graph_builder.py       # 构建叙事结构图谱（事件因果、情节单元）
│   ├── graph_preprocessor.py            # 实体消歧、聚类与合并
│   ├── event_causality_builder.py       # 事件因果关系构建与图分析
|   ├── graph_analyzer.py                # 图相关的函数调用
|   ├── document_parser.py               # 文字解析类的函数调用
│   ├── knowledge_extractor.py           # 知识图谱抽取相关的函数调用
│   └── reflection.py                    # 反思相关的功能
|
├── functions/
│   └── regular_functions/
│       ├── entity_extraction.py         # 实体抽取工具函数
│       ├── relation_extraction.py       # 关系抽取工具函数
│       ├── attribute_extraction.py      # 属性抽取工具函数
│       └── reflect_extraction.py        # 抽取结果质量评分与反思
│
├── llm/
│   ├── llm_manager.py                    # LLM 初始化与统一接口管理
│   └── qwen3_llm.py                      # Qwen 本地模型加载、推理封装
│
├── storage/
│   ├── vector_store.py                  # 基于 Chroma 的语义向量存储
│   └── graph_store.py                    # 图谱存储与查询封装
│
└── utils/
    ├── neo4j_utils.py                    # Neo4j 操作封装
    ├── prompt_loader.py                  # Prompt 模板加载
    ├── format.py                         # JSON 修复、格式化
    └── config.py                         # 配置文件解析
```

---

## 核心运行流程（基于 `main.py`）

1. **加载配置**（`-c` 指定）

   * 初始化 LLM（OpenAI / Qwen）
   * 连接 Neo4j
   * 加载构建设置（`-b` 指定）
   * embedding模型最好使用bge，并在config里面设置好路径

2. **文档处理**

   * 加载 JSON 文档（`-i` 指定）
   * 文本切分（多线程）
   * 生成 chunk 元数据（章节/场景顺序等）

3. **知识抽取**

   * 调用 `KGExtractionAgent` 抽取实体与关系
   * 调用 `AttributeExtractionAgent` 抽取属性
   * 抽取结果经过 `GraphPreprocessor` 消歧与合并

4. **图谱构建**

   * 写入实体与关系到 Neo4j
   * 生成并写入向量嵌入

5. **可选分析步骤**

   * 构建事件因果图谱
   * 聚合情节单元图谱

---

## 输出内容

* **Neo4j 节点与关系**

  * 节点类型：Entity / Event / Plot / Scene（按 doc\_type 不同可能变化）
  * 关系类型：多种事件关系、Plot 内部关系等
* **向量索引**（Neo4j 内部向量索引）
* **本地文件输出**

  * 抽取结果 JSON（带实体、关系、属性）
  * 日志文件（verbose 模式）

---

## 依赖环境

* Python 3.10+
* Neo4j 5.x
* LangChain / LangGraph
* Transformers / vLLM
* SentenceTransformers
* ChromaDB

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 安装 Neo4j

```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装依赖
sudo apt install wget apt-transport-https gnupg lsb-release -y

# 3. 添加 Neo4j 官方 GPG key
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg

# 4. 添加 Neo4j 源（以 Neo4j 5.x 为例）
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list

# 5. 安装 Neo4j
sudo apt update
sudo apt install neo4j -y

# 6. 启动服务并设置开机自启
sudo systemctl enable neo4j
sudo systemctl start neo4j

# 7. 查看运行状态
sudo systemctl status neo4j

# 8. 安装 GDS
cp neo4j-graph-data-science-2.13.4.jar /var/lib/neo4j/plugins/

# 9. 在 conf/neo4j.conf 中添加表头：
nano /etc/neo4j/neo4j.conf
# 然后添加：
dbms.security.procedures.unrestricted=gds.*
dbms.security.procedures.allowlist=gds.*

```


---

## 开发建议

* 长文档建议多进程 + 多 GPU 并行
* 对低置信度抽取结果启用反思重抽
* 使用 Neo4j Bloom / GraphXR 可视化探索
