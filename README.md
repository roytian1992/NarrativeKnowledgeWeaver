# KGAG - Knowledge Graph and Agent Generation

## 项目简介

KGAG (Knowledge Graph and Agent Generation) 是一个基于大语言模型的智能知识图谱构建和事件因果关系分析系统。该项目结合了现代自然语言处理技术和图数据库技术，能够从非结构化文本中自动提取实体、关系和属性，构建结构化的知识图谱，并进一步分析事件之间的因果关系。

## 核心特性

### 🧠 智能知识图谱构建
- **自动实体提取**: 使用大语言模型从文本中识别和提取实体
- **关系识别**: 智能识别实体间的语义关系
- **属性提取**: 自动提取实体的详细属性信息
- **多模态支持**: 支持多种文本格式的处理

### 🔗 事件因果关系分析
- **因果链发现**: 自动识别事件之间的因果关系
- **时序分析**: 基于时间序列的事件关联分析
- **图谱可视化**: 提供直观的因果关系图谱展示

### 🚀 基于LangChain的Agent架构
- **模块化设计**: 基于LangChain框架的标准化开发
- **记忆机制**: 支持任务记忆和反思机制
- **可扩展性**: 易于扩展和定制的Agent系统

## 系统架构

```
KGAG/
├── kag/                    # 核心代码模块
│   ├── agent/             # Agent代理模块
│   ├── builder/           # 构建器模块（核心）
│   │   ├── kg_builder.py          # 知识图谱构建器
│   │   ├── event_causality_builder.py  # 事件因果关系构建器
│   │   ├── extractor.py           # 信息提取器
│   │   ├── processor.py           # 文档处理器
│   │   └── reflection.py          # 反思模块
│   ├── functions/         # 功能模块
│   ├── llm/              # 大语言模型模块
│   ├── memory/           # 记忆模块
│   ├── models/           # 数据模型
│   ├── prompts/          # 提示词模板
│   ├── reflection/       # 反思机制
│   ├── schema/           # 模式定义
│   ├── storage/          # 存储模块
│   └── utils/            # 工具模块
├── configs/              # 配置文件
├── main.py              # 主程序入口
└── requirements.txt     # 依赖包列表
```


## 环境要求

- Python 3.8+
- Neo4j 数据库
- OpenAI API 密钥（或其他兼容的LLM服务）

## 安装说明

### 1. 克隆项目
```bash
git clone https://github.com/roytian1992/KGAG.git
cd KGAG
```

### 2. 创建虚拟环境
```bash
# 使用conda
conda env create -f environment.yml
conda activate kgag

# 或使用pip
pip install -r requirements.txt
```

### 3. 配置环境
复制并编辑配置文件：
```bash
cp configs/config_openai.yaml configs/config.yaml
```

在配置文件中设置：
- OpenAI API密钥
- Neo4j数据库连接信息
- 其他必要的配置参数

## 主要依赖

### 核心框架
- **LangChain**: 大语言模型应用开发框架
- **LangGraph**: 图结构的Agent工作流
- **LangChain-Community**: 社区扩展组件

### 数据存储
- **Neo4j**: 图数据库，用于存储知识图谱
- **ChromaDB**: 向量数据库，用于语义搜索
- **Sentence-Transformers**: 文本向量化

### 机器学习
- **Transformers**: Hugging Face变换器模型
- **PyTorch**: 深度学习框架
- **OpenAI**: OpenAI API客户端

## 快速开始

### 基本使用流程

1. **准备数据**: 将待处理的文本数据保存为JSON格式
2. **配置系统**: 根据需要修改配置文件
3. **运行构建**: 执行主程序构建知识图谱
4. **分析结果**: 查看生成的知识图谱和因果关系

### 命令行使用

```bash
python main.py --input your_data.json --config configs/config.yaml --verbose
```

### 参数说明

- `--input`: 输入JSON文件路径（必需）
- `--config`: 配置文件路径（默认: configs/default.yaml）
- `--verbose`: 启用详细输出模式
- `--output-stats`: 输出统计信息


## 核心功能详解

### 知识图谱构建流程

系统采用模块化的构建流程，主要包括以下步骤：

```python
# 创建知识图谱构建器
builder = KnowledgeGraphBuilder(config)

# 1. 准备文本块
builder.prepare_chunks(args.input, verbose=args.verbose)

# 2. 存储文本块
builder.store_chunks(verbose=args.verbose)

# 3. 提取实体和关系
builder.extract_entity_and_relation(verbose=args.verbose)

# 4. 提取实体属性
builder.extract_entity_attributes(verbose=args.verbose)

# 5. 构建知识图谱
kg = builder.build_graph_from_results(verbose=args.verbose)

# 6. 构建事件因果关系图
event_graph_builder = EventCausalityBuilder(config)
causality_graph = event_graph_builder.build_complete_causality_graph()
```

### 1. 文本预处理 (`prepare_chunks`)
- 将输入文档分割成适当大小的文本块
- 保持语义完整性的同时优化处理效率
- 支持多种文档格式和编码

### 2. 文本存储 (`store_chunks`)
- 将处理后的文本块存储到向量数据库
- 建立文本的向量索引以支持语义搜索
- 优化存储结构以提高查询性能

### 3. 实体关系提取 (`extract_entity_and_relation`)
- 使用大语言模型识别文本中的实体
- 分析实体间的语义关系
- 应用预定义的本体模式进行标准化

### 4. 属性提取 (`extract_entity_attributes`)
- 为识别的实体提取详细属性信息
- 丰富实体的语义描述
- 支持多层次的属性结构

### 5. 图谱构建 (`build_graph_from_results`)
- 将提取的实体、关系和属性整合成知识图谱
- 建立图数据库中的节点和边
- 优化图结构以支持高效查询

### 6. 因果关系分析 (`build_complete_causality_graph`)
- 识别事件之间的因果关系
- 构建有向的因果关系图
- 支持时序分析和因果链追踪

## 输出结果

系统运行完成后会输出以下统计信息：

- **实体数量**: 识别的实体总数
- **关系数量**: 提取的关系总数  
- **文档数量**: 处理的文档总数
- **文本块数量**: 分割的文本块总数

## 配置说明

### 配置文件结构

项目提供两种配置模板：

- `config_local.yaml`: 本地开发环境配置
- `config_openai.yaml`: OpenAI服务配置

### 主要配置项

```yaml
# LLM配置
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "your-api-key"

# 数据库配置
database:
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
  
  vector_store:
    provider: "chromadb"
    collection_name: "kgag_vectors"

# 处理配置
processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_workers: 16
```


## 使用示例

### 示例数据格式

输入的JSON文件应包含待处理的文本数据：

```json
{
  "documents": [
    {
      "id": "doc_001",
      "title": "示例文档标题",
      "content": "这里是文档的主要内容...",
      "metadata": {
        "source": "example_source",
        "date": "2024-01-01"
      }
    }
  ]
}
```

### 完整使用示例

```python
from kag.builder.kg_builder import KnowledgeGraphBuilder
from kag.builder.event_causality_builder import EventCausalityBuilder
from kag.utils.config import KAGConfig

# 加载配置
config = KAGConfig.from_yaml("configs/config.yaml")

# 创建知识图谱构建器
builder = KnowledgeGraphBuilder(config)

# 执行完整的构建流程
try:
    # 数据预处理
    print("🔄 准备文本块...")
    builder.prepare_chunks("data/input.json", verbose=True)
    
    print("💾 存储文本块...")
    builder.store_chunks(verbose=True)
    
    # 知识提取
    print("🧠 提取实体和关系...")
    builder.extract_entity_and_relation(verbose=True)
    
    print("📝 提取实体属性...")
    builder.extract_entity_attributes(verbose=True)
    
    # 构建图谱
    print("🔗 构建知识图谱...")
    kg = builder.build_graph_from_results(verbose=True)
    
    # 因果关系分析
    print("⚡ 构建因果关系图...")
    event_builder = EventCausalityBuilder(config)
    causality_graph = event_builder.build_complete_causality_graph()
    
    # 输出统计信息
    stats = builder.get_stats()
    print(f"✅ 构建完成!")
    print(f"   - 实体数量: {stats['entities']}")
    print(f"   - 关系数量: {stats['relations']}")
    print(f"   - 文档数量: {stats['documents']}")
    
except Exception as e:
    print(f"❌ 构建失败: {str(e)}")
```

## 技术特点

### 🎯 基于LangChain的标准化架构
- 采用LangChain框架进行开发，确保代码的标准化和可维护性
- 支持多种LLM提供商（OpenAI、Anthropic等）
- 模块化设计便于扩展和定制

### 🧠 智能Agent系统
- **任务管理**: 支持复杂任务的规划、状态追踪和错误纠正
- **记忆机制**: 实现任务记忆和经验总结
- **反思能力**: 具备策略优化和自我改进能力

### 📊 多模态数据处理
- 支持文本、结构化数据等多种输入格式
- 灵活的数据预处理和后处理管道
- 可配置的文本分块和向量化策略

### 🔍 高效的图数据管理
- 基于Neo4j的高性能图数据库存储
- 支持复杂的图查询和分析
- 优化的图结构设计

### 🚀 可扩展的处理能力
- 支持多线程并行处理
- 可配置的批处理大小
- 内存优化的大规模数据处理

## 项目结构详解

### 核心模块说明

#### `kag/builder/` - 构建器模块
- **kg_builder.py**: 知识图谱构建的核心逻辑
- **event_causality_builder.py**: 事件因果关系分析
- **extractor.py**: 信息提取器，负责从文本中提取结构化信息
- **processor.py**: 文档处理器，处理各种格式的输入数据
- **reflection.py**: 反思模块，提供自我评估和改进能力

#### `kag/agent/` - Agent代理模块
- 实现基于LangChain的智能代理
- 支持复杂任务的自动化执行
- 提供任务规划和状态管理

#### `kag/memory/` - 记忆模块
- 实现任务记忆和经验存储
- 支持长期记忆和短期记忆
- 提供记忆检索和更新机制

#### `kag/storage/` - 存储模块
- 图数据库接口封装
- 向量数据库管理
- 文档存储和检索

## 性能优化

### 处理效率优化
- **并行处理**: 支持多线程并行处理大规模数据
- **批处理**: 优化的批处理策略减少API调用次数
- **缓存机制**: 智能缓存减少重复计算

### 内存管理
- **流式处理**: 支持大文件的流式处理
- **内存监控**: 实时监控内存使用情况
- **垃圾回收**: 优化的内存回收策略


## 常见问题 (FAQ)

### Q: 支持哪些大语言模型？
A: 项目基于LangChain框架，支持多种LLM提供商：
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- 本地部署的开源模型
- 其他兼容OpenAI API的服务

### Q: 如何处理大规模数据？
A: 系统提供多种优化策略：
- 调整`max_workers`参数增加并行度
- 使用适当的`chunk_size`分割大文档
- 启用流式处理模式
- 配置合适的批处理大小

### Q: 如何自定义实体和关系类型？
A: 可以通过以下方式自定义：
- 修改`kag/schema/`中的模式定义文件
- 更新提示词模板以适应特定领域
- 扩展实体和关系的提取逻辑

### Q: 系统的准确性如何？
A: 准确性取决于多个因素：
- 使用的LLM模型质量
- 提示词的设计质量
- 输入数据的质量和结构
- 领域特定的配置优化

## 故障排除

### 常见错误及解决方案

#### 1. Neo4j连接失败
```bash
# 检查Neo4j服务状态
sudo systemctl status neo4j

# 重启Neo4j服务
sudo systemctl restart neo4j
```

#### 2. OpenAI API限制
- 检查API密钥是否有效
- 确认账户余额充足
- 调整请求频率以避免限制

#### 3. 内存不足
- 减少`chunk_size`参数
- 降低`max_workers`数量
- 启用流式处理模式

## 开发指南

### 本地开发环境搭建

1. **安装Neo4j**
```bash
# Ubuntu/Debian
sudo apt-get install neo4j

# macOS
brew install neo4j
```

2. **配置开发环境**
```bash
# 复制开发配置
cp configs/config_local.yaml configs/config.yaml

# 安装开发依赖
pip install -r requirements-dev.txt
```

3. **运行测试**
```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python -m pytest tests/integration/
```

### 代码贡献

我们欢迎社区贡献！请遵循以下步骤：

1. **Fork项目**并创建特性分支
2. **编写代码**并确保通过所有测试
3. **更新文档**以反映您的更改
4. **提交Pull Request**并详细描述更改内容

### 代码规范

- 遵循PEP 8 Python代码规范
- 使用类型注解提高代码可读性
- 编写完整的文档字符串
- 保持测试覆盖率在80%以上

## 路线图

### 近期计划 (v1.1)
- [ ] 支持更多文档格式（PDF、Word等）
- [ ] 增加图谱可视化界面
- [ ] 优化大规模数据处理性能
- [ ] 添加更多预训练的领域模型

### 中期计划 (v1.5)
- [ ] 支持多语言处理
- [ ] 实现增量更新机制
- [ ] 添加图谱质量评估工具
- [ ] 集成更多向量数据库

### 长期计划 (v2.0)
- [ ] 支持多模态数据（图像、音频）
- [ ] 实现联邦学习能力
- [ ] 添加自动化模型优化
- [ ] 构建完整的知识图谱生态系统

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 致谢

感谢以下开源项目的支持：
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- [Neo4j](https://neo4j.com/) - 图数据库
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Transformers](https://github.com/huggingface/transformers) - 预训练模型库

## 联系方式

- **项目维护者**: roytian1992
- **GitHub**: https://github.com/roytian1992/KGAG
- **问题反馈**: 请通过GitHub Issues提交

---

如果这个项目对您有帮助，请给我们一个⭐️！

