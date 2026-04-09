# NarrativeKnowledgeWeaver Agent Runtime Workflow Report

## 1. 目标

本文档面向方法与流程层面的分析，回答三个问题：

1. 旧版 `qwen-agent` 在本项目中的真实工作流程是什么。
2. 我们最早实现的 `langgraph` 版内部是如何运行的。
3. 当前版本在内部执行链上与前两者相比具体改了什么。

重点不是列代码接口，而是刻画一次问答从输入问题到产出答案时，系统内部到底经历了哪些阶段，谁负责选工具，谁负责执行工具，工具结果如何回流到下一轮推理。


## 2. 统一外层框架

无论是旧版 `qwen-agent`，还是后来的 `langgraph` 版，外层问答框架其实高度一致。真正变化最大的部分，不是 `ask()` 外层，而是单路回答时内部 assistant runtime 的实现方式。

一次问答的统一外层流程可以抽象为：

1. 接收问题 `user_text`。
2. 构造 `memory_ctx`。
   - 包括策略记忆读取结果。
   - 包括 `query_pattern`、候选模板、可选的 `routing_hint`。
3. 决定本题是否进入特殊模式：
   - `online runtime matching`
   - `offline/online strategy subagent`
   - `sampling branches`
   - 否则进入单路回答 `single_route`
4. 在 `single_route` 中：
   - 构造运行时 system message
   - 决定当前 assistant 能看到哪些工具
   - 把问题交给一个具体 assistant runtime
5. assistant runtime 内部多轮运行，直到：
   - 不再请求工具
   - 或达到内部预算上限
6. 外层拿到完整消息轨迹后，再做答案提取、adapter、可选的 online 写回。

因此，三种版本真正的差别，主要集中在第 4 步和第 5 步之间。


## 3. 旧版 Qwen-Agent 工作流

### 3.1 我们项目里的接入方式

旧版入口在 `core/agent/retriever_agent_qwen.py`。

单题执行时：

1. `ask()` 先准备 `memory_ctx`，并判断是否需要：
   - 在线模板命中
   - 采样分支
   - subagent 分支
2. 如果走单路回答，则进入 `_run_single_route(...)`。
3. `_run_single_route(...)` 会：
   - 构造 `runtime_system_message`
   - 构造 `tool_subsets`
   - 为当前工具子集创建一个 `qwen_agent.agents.assistant.Assistant`
   - 调用 `assistant.run_nonstream(...)`
4. 一旦某个 stage 已经产生工具调用或文本回答，就结束并返回。

这里要注意一件事：

- 在我们项目里，`qwen` 版本默认常常走 `qwen_native_tool_routing_only=true`。
- 这意味着 `_build_tool_execution_plan()` 直接返回 `[self._all_tools()]`。
- 换言之，外层通常不再替它筛工具，而是把全部可见工具直接交给 Qwen 自己决定。

所以旧版的核心特征是：

- 外层负责模式选择与分支管理。
- 内层具体的工具路由与工具链展开，主要由 Qwen 原生 assistant 自己完成。

### 3.2 Qwen Assistant 的内部结构

`qwen_agent.agents.assistant.Assistant` 不是一个单独的大黑箱，它的继承链是：

`Assistant -> FnCallAgent -> Agent`

这三层分别承担不同职责。

#### 第 1 层：Agent

`Agent` 是统一基类，负责：

1. 标准化输入消息格式。
2. 自动插入 system message。
3. 调用底层 `llm.chat(...)`。
4. 执行 `_call_tool(...)`。
5. 识别模型输出里是否包含 `function_call`。

这一层本身并不定义“多轮工具链”，只提供：

- 调 LLM
- 调工具
- 把消息转成统一结构

#### 第 2 层：FnCallAgent

真正的工具循环在 `FnCallAgent._run(...)` 中。

它的逻辑可以概括为：

1. 复制消息，初始化 `num_llm_calls_available = MAX_LLM_CALL_PER_RUN`。
2. 进入 while 循环，只要还有 LLM 调用预算就继续。
3. 每一轮调用 `_call_llm(...)`，并把所有工具 schema 作为 `functions` 传给模型。
4. 收集模型返回的 assistant 消息。
5. 遍历这些输出，检查其中是否有 `function_call`。
6. 对每个 `function_call`：
   - 解析 `tool_name`
   - 解析 `tool_args`
   - 执行工具
   - 构造一条 `FUNCTION` 消息，把工具结果追加回消息历史
7. 如果本轮用了任意工具，则继续下一轮 LLM。
8. 如果本轮没有工具调用，则认为已经给出最终回答，循环结束。

这实际上就是一个标准的 function-calling agent loop：

`LLM -> detect function call -> run tool -> append function result -> LLM -> ...`

#### 第 3 层：Assistant

`Assistant` 在 `FnCallAgent` 外面又包了一层轻量 RAG 包装。

其工作是：

1. 在 `_run(...)` 前先执行 `_prepend_knowledge_prompt(...)`。
2. 如果显式给了 `knowledge`，直接使用。
3. 否则通过 `self.mem.run(...)` 从 agent 的 file memory 中检索内容。
4. 把得到的知识片段整理成知识库提示块，拼到 system prompt 前面。
5. 再把新消息交回 `FnCallAgent._run(...)`。

因此，严格来说，Qwen Assistant 的完整流程是：

`Question -> optional file-memory retrieval -> prepend knowledge prompt -> FnCallAgent multi-turn function calling loop`

不过在我们项目里，这个“文件型 RAG”能力并不是主角。因为我们主要依赖显式注册的检索工具来访问图、向量库和 SQL，而不是依赖 Assistant 自己从上传文件里做检索。

### 3.3 Qwen 的 function-calling 提示格式

Qwen 并不是简单把工具 schema 扔给模型就结束了，它还有一层专门的 function-calling prompt 适配。

`qwen_fncall_prompt.py` 会做两件关键事情：

1. **预处理**
   - 把历史上的 `function_call` / `FUNCTION result` 重写为带特殊标记的纯文本片段。
   - 插入工具说明块。
   - 告诉模型可以输出：
     - `✿FUNCTION✿`
     - `✿ARGS✿`
     - `✿RESULT✿`
     - `✿RETURN✿`

2. **后处理**
   - 再把模型输出的这套文本标记解析回结构化 `function_call`。
   - 如果 `parallel_function_calls=False`，只保留第一个工具调用。
   - 如果允许并行，则一条回复里可以解析出多个工具调用。

也就是说，Qwen Assistant 的“工具调用”本质上并不只是 OpenAI-style JSON tool call，它还有一层自己的 prompt grammar 和 message rewriting。

### 3.4 Qwen 版本的一次典型问答轨迹

一个单题流程可写成：

1. `ask(question)`
2. 读取策略记忆，决定是否分支
3. 若单路：
   - 构造 `runtime_system_message`
   - 默认把全部可见工具交给 `Assistant`
4. `Assistant._run`
   - 可选地从 file memory 拼接知识块
5. `FnCallAgent._run`
   - LLM 第 1 轮决定是否调用工具
   - 若调用，则执行工具并追加 `FUNCTION` 消息
   - LLM 第 2 轮基于工具结果继续判断
   - 重复直到不再调用工具
6. 外层抽取最终回答

因此，旧版 `qwen-agent` 的核心优势，不在外层问答控制，而在于：

- 它把工具循环、函数调用格式、工具结果回流都交给了一个较成熟的原生 function-calling runtime。
- 在默认情况下，它更少依赖外部硬路由，更依赖模型自己在全工具集合里做选择。


## 4. 早期 LangGraph 版工作流

### 4.1 总体定位

早期 `langgraph` 版的外层流程与 `qwen` 版几乎同构，但内层 assistant 被替换成了我们自己的 `LangGraphAssistantRuntime`。

因此它的核心结构变成：

`QuestionAnsweringAgent.ask -> _run_single_route -> LangGraphAssistantRuntime.run_nonstream`

### 4.2 外层与 Qwen 相同的部分

在 `retriever_agent_langchain.py` 中，外层仍然会：

1. 构造 `memory_ctx`
2. 可选注入 `routing_hint`
3. 决定是否触发：
   - online strategy
   - sampling branch
   - subagent
4. 若进入单路模式，则：
   - 生成 runtime system message
   - 调 `tool_router.build_tool_execution_plan(...)`
   - 为每个工具子集创建一个 `LangGraphAssistantRuntime`

因此外层“长什么样”，并不是早期 langgraph 弱于 qwen 的根本原因。差异主要来自内部 runtime。

### 4.3 旧版 LangGraphAssistantRuntime 的内部机制

早期 runtime 不是原生 function calling，而是一个**手写的 JSON-only tool loop**。

它的核心思想是：

1. 给模型一个 system prompt，明确要求它只能输出两种 JSON：
   - `{"tool_name": "...", "tool_arguments": {...}}`
   - `{"final_answer": "..."}`
2. 如果模型输出工具调用 JSON，则 runtime 自己把它转换成 `AIMessage.tool_calls`
3. LangGraph 的状态图执行：
   - `model`
   - `tools`
   - `model`
   - ...
4. 工具结果以 `ToolMessage` 的形式回流给下一轮模型
5. 直到模型不再要求工具，流程结束

旧版状态图本质上是：

`model -> if tool_call then tools -> append ToolMessage -> model -> ...`

### 4.4 旧版 LangGraph 的具体特点

旧版 runtime 的关键性质有三点：

1. **单轮最多一个工具**
   - prompt 明确要求 `Call at most one tool per turn.`
   - 这意味着它天然是串行链式，而非并行工具调用。

2. **没有总工具预算**
   - 它有 `remaining_llm_calls`
   - 但没有 `remaining_tool_calls`
   - 只要模型持续要求工具，且没耗尽 LLM 轮数，就能继续调

3. **消息回流是手动摘要式的**
   - runtime 不依赖底层模型 endpoint 的原生 tool calling
   - 而是自己构造“Conversation so far”文本，把历史对话、工具调用、工具结果重新组织给模型看

因此，旧版 langgraph 的内部循环更像：

`manual JSON ReAct`

而不是：

- Qwen 那种 native function-calling loop
- 也不是完全自由文本 ReAct

### 4.5 旧版 LangGraph 的流程图

一次单题执行流程可写成：

1. 外层 `ask()` 决定是否走单路
2. 外层 router 决定本轮 assistant 可见哪些工具
3. `LangGraphAssistantRuntime.run_nonstream(...)`
4. 初始化状态：
   - `messages`
   - `remaining_llm_calls = 8`（默认）
5. `model` 节点：
   - 把历史 transcript 和工具列表转成 prompt
   - 要求模型输出 JSON
6. 如果输出是工具调用：
   - 进入 `tools` 节点
   - 执行工具
   - 把结果作为 `ToolMessage` 追加
   - 回到 `model`
7. 如果输出是最终答案：
   - 结束

它的优点是可控，容易兼容现有工具协议。

它的缺点是：

- 所有工具调用逻辑都靠我们自己的 JSON prompt 约束
- 不像 qwen 那样天然继承成熟的原生 function-calling 行为
- 缺少总工具预算时，模型容易在多轮里做过多探索


## 5. 当前 LangGraph 版工作流

### 5.1 变化范围

当前版本的外层框架并没有根本改写，真正的变化集中在 `LangGraphAssistantRuntime` 内部。

这次改动的目标是把原先“可能拖很多轮”的 manual JSON loop，收紧成更短、更像显式证据链的工具调用过程。

### 5.2 当前版本的核心约束

当前 runtime 在旧版基础上新增了 `remaining_tool_calls`，并把总工具调用数限制为 3。

因此当前单题执行的硬约束变成：

1. 单轮最多 1 个工具。
2. 整道题最多 3 次工具调用。
3. 仍然最多若干次 LLM 轮数，默认是 8。
4. 当工具预算耗尽后，不允许继续调工具，只能基于已有证据直接回答。

换言之，当前版本强调的是：

`A -> B -> C`

这样的短证据链，而不是无限制地继续试错。

### 5.3 当前版本的内部流程

当前 `LangGraphAssistantRuntime` 的运行流程如下：

1. 初始化状态：
   - `messages`
   - `remaining_llm_calls`
   - `remaining_tool_calls`

2. `model` 节点读取当前消息历史，并构造一个新的“Conversation so far”提示。

3. system prompt 里明确加入三条关键规则：
   - 一轮至多一个工具
   - 全题至多 3 个工具
   - 如果需要，构造一个短链式证据路径，而不是同时调很多工具

4. 模型输出后，runtime 解析：
   - 如果是 `tool_name/tool_arguments`
     - 转成 `AIMessage.tool_calls`
   - 如果是 `final_answer`
     - 转成普通 assistant 消息

5. 如果模型请求工具，且 `remaining_tool_calls > 0`：
   - 进入 `tools` 节点
   - 执行工具
   - 生成 `ToolMessage`
   - `remaining_tool_calls -= 1`
   - 回到 `model`

6. 如果模型请求工具，但预算已经为 0：
   - runtime 会拦截这次工具请求
   - 强制改写为“必须基于现有证据直接作答”

7. 如果模型不再请求工具，则结束。

### 5.4 当前版本相对于旧版 LangGraph 的本质变化

当前版本和旧版相比，内部语义上有三个实质变化：

1. **从“仅限制每轮一个工具”变成“整题总工具预算受限”**
   - 旧版只限制单轮
   - 当前版还限制总步数

2. **从“可拖长探索”变成“鼓励短链决策”**
   - 旧版更像松散的多轮 JSON agent
   - 当前版更像受限 ReAct

3. **从“模型可以一直再试一个工具”变成“预算耗尽后必须收束”**
   - 这直接提高了执行可控性
   - 但也可能减少探索深度


## 6. 三者并列比较

### 6.1 决策权分布

| 维度 | 旧版 Qwen-Agent | 旧版 LangGraph | 当前 LangGraph |
|---|---|---|---|
| 外层策略记忆/分支 | 我们控制 | 我们控制 | 我们控制 |
| 内层工具路由 | 主要由 Qwen 原生 Assistant | 由我们手写 JSON tool loop + 模型决定 | 由我们手写 JSON tool loop + 模型决定 |
| 工具可见性 | 常常直接全量工具 | 常由 tool router 先裁剪 | 常由 tool router 先裁剪 |
| 单轮工具数 | 可多于 1，取决于 function-calling 配置 | 1 | 1 |
| 整题工具预算 | 仅受 `MAX_LLM_CALL_PER_RUN` 间接限制 | 无显式工具预算 | 显式限制为 3 |

### 6.2 对工具链形态的影响

#### 旧版 Qwen-Agent

- 更接近原生 function-calling assistant
- 一次回复中理论上可以产生多个工具调用
- 工具结果回流形式更贴近框架原生消息结构
- 探索空间更大，但控制粒度更弱

#### 旧版 LangGraph

- 明确是一轮一个工具
- 但整题没有总工具步数上限
- 更容易出现“继续再试一个工具”的长尾行为

#### 当前 LangGraph

- 仍是一轮一个工具
- 但强制整题最多 3 工具
- 更适合构造短证据链
- 更适合你想要的 `A -> B -> C` 形式


## 7. 为什么会出现性能差异

从运行机制上看，旧版 `qwen-agent` 之所以经常显得“更会用工具”，并不一定是因为它外层框架更高级，而更可能是因为：

1. 它把更大一部分决策权交给了成熟的原生 function-calling runtime。
2. 它默认往往直接给全量工具，而不是先被外部 router 裁掉。
3. 它允许更自然地连续扩展工具调用，而不是完全依赖我们设计的 JSON prompt 约束。

而 `langgraph` 版本的优势是：

1. 我们可以精细控制工具预算。
2. 我们可以清晰规定链式调用形态。
3. 我们可以更方便接入自己的策略记忆、adapter、fallback 和后处理。

所以二者的根本差异，不是“是否有 agent”，而是：

- Qwen 更偏原生 function-calling assistant
- LangGraph 更偏自定义可控 runtime


## 8. 结论

可以把三种版本概括为以下三句话。

### 8.1 旧版 Qwen-Agent

外层由我们决定是否读策略记忆、是否分支；一旦进入单路回答，内部的工具规划、工具切换、继续探索与停止，主要交给 Qwen 原生 `Assistant/FnCallAgent` 处理。

### 8.2 旧版 LangGraph

外层仍然沿用我们的策略框架，但内部 assistant 被替换成手写的 JSON-only tool loop；它已经具备串行链式工具调用能力，但缺少总工具预算，因此探索深度较松。

### 8.3 当前 LangGraph

当前版本保留了手写 runtime 的可控性，同时加入总工具预算和短链约束，使内部执行更接近“受限 ReAct”：每轮一个工具、整题最多三步、工具结果逐轮回灌，强制形成短证据链并在预算耗尽时收束作答。


## 9. 对后续设计的启示

如果目标是同时保留探索能力与可控性，那么最值得单独优化的部分，不再是外层 `ask()`，而是内部 assistant runtime 本身。具体地说，未来可以优先考虑：

1. 是否保留“每轮一个工具”的串行链结构。
2. 是否把总工具预算从固定 3 改成按题型自适应。
3. 是否保留外部 router，还是把更多决策重新交还给内部 runtime。
4. 如何让内部 runtime 学会在“探索”与“收束”之间切换，而不是一味缩短或一味扩张工具链。

这也是为什么，在本项目当前阶段，理解并重构 internal tool loop，比继续堆叠外部 routing hint 更关键。
