# 2026-04-14 NarrativeKnowledgeWeaver LangGraph Handoff

## 1. 当前目标与状态

当前主线已经从早期的 QUALITY 路由/策略试验，转到两条并行工作：

1. 维护当前锚定的 `no_strategy_agent` / LangGraph agent 主线。
2. 把同一套知识抽取与问答流程迁移到 `FiarytableQA` 开放问答基准。

截至 2026-04-14 晚间，`FiarytableQA` 的 100 篇实验仍在运行中，尚未生成最终 `summary.json`。

当前部分进度：

- 已完成文章数：`92 / 100`
- 已完成题目数：`3718`
- 当前部分指标：
  - `overall_accuracy = 0.6718`
  - `pass_accuracy = 0.9021`
  - `avg_judge_score = 0.6942`
  - `avg_latency_ms = 26359.61`
- 当前 progress 文件：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/reports/progress.json`
- 当前 tmux 会话：
  - `fiarytableqa100_20260412`


## 2. 关键结论

### 2.1 QUALITY 侧的历史锚点

当前本地对历史实验名的唯一可信解释，已经写入：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/artifacts/regression/run_alias_notes.md`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/artifacts/regression/run_aliases.json`

其中最重要的是：

- `s2`
  - 本地唯一规范含义：`2026-04-10` 重建出的 50 篇 canonical 子集
  - summary：
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/quality/artifacts/regression/s2_canonical_summary_20260410.json`
  - 指标：
    - `overall = 0.6099`
    - `pass = 0.8377`
    - `article_count = 50`
    - `question_count = 770`

- `s1_recover`
  - 历史强基线，可恢复
  - run：
    - `quality_routerllm_remaining102_20260404_s1`
  - 指标：
    - `overall = 0.6326`
    - `pass = 0.8847`
    - `question_count = 399`

后续如果再提 `s2`，应只指 `s2_canonical_summary_20260410.json` 对应的那版，不再使用旧的含混说法。

### 2.2 当前 FiarytableQA 方向

这个方向已经不是选择题实验，而是开放问答：

- 不再使用 MCQ adapter
- 改为普通 open-answer adapter
- 结果由 `llm-as-judge` 判定
- 支持多 GT 答案同时作为 judge reference
- 抽取完成的文章 workspace 保存在当前项目下，便于后续直接加载


## 3. 这轮做过的关键改动

### 3.1 FiarytableQA benchmark runner 已经落地

新增目录：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa`

核心 runner：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/run_fiarytable_benchmark.py`

这个 runner 复用了 QUALITY 的大部分流水线能力，包括：

- article workspace 准备
- 抽取后复用 workspace / graph / sql / vector store
- 多次 repeat 评测
- per-article JSON report 落盘
- 最终 summary 汇总

但做了几个数据集专用改动：

- 数据集输入从 `articles/*.json + questions/*.csv` 读取
- 每题可有多个 reference answers
- 增加 `question_type_fields` / `question_type_tags`
- summary 设计上支持按问题类型做 breakdown

### 3.2 FiarytableQA 采用 single-chunk profile

扫描后确认，这个数据集每个原始 `content` 段都没有超过 `800` 个英文词，最大约 `446`。

因此当前实现没有改动全局 `DocumentProcessor`，而是在 Fiarytable runner 中做了数据集级配置覆盖，使其等价于：

- 每个原始 `content` 作为单个 chunk
- 不走原本对长文本的 chunk+merge 拆分
- 但继续保留 summary / metadata 的产出格式一致性

相关文件：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/run_fiarytable_benchmark.py`

### 3.3 开放问答 judge prompt 已落地

新增 prompt：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/task_specs/prompts_en/memory/judge_open_retrieval_answer.yaml`

用途：

- 对开放问答结果做 `llm-as-judge`
- 容纳多个 GT answer variants
- 判分比 MCQ 更宽松，避免只因表述差异误判

### 3.4 hidden tools 的实际过滤 bug 已修

修复点：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent_langchain.py`

关键修复：

- `_rebuild_assistant()` 现在使用 `_all_tools()`
- 因而 `hidden_tool_names` 终于会真实影响 agent 可见工具集

这个修复对 FiarytableQA 很关键，因为该数据集明确不应暴露 MCQ 专用工具。

### 3.5 FiarytableQA 中显式隐藏了 MCQ 工具

当前 Fiarytable runner 会隐藏：

- `choice_grounded_evidence_search`

相关实现位于：

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/run_fiarytable_benchmark.py`


## 4. 当前最重要的路径

### 4.1 仓库根目录

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph`

### 4.2 当前 FiarytableQA 运行目录

- run 根目录：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412`
- manifest：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/manifest.json`
- reports：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/reports`
- log：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/logs/fiarytableqa_no_strategy_100_20260412.log`

### 4.3 FiarytableQA 抽取产物

- article workspaces：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/assets/article_workspaces`
- converted articles：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/assets/converted_articles`

### 4.4 数据集原始路径

- FiarytableQA：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/FiarytableQA`
- QUALITY：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/SCROLLS/quality/quality_processed`

### 4.5 运行环境

- conda env：
  - `screenplay`
- 激活命令：
  - `source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay`
- 长任务启动方式：
  - 用 `tmux`
  - 不再用 `nohup`


## 5. 当前 agent / runtime 的重要代码位置

### 5.1 主 agent

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retriever_agent_langchain.py`

这个文件负责：

- 问答主入口
- memory / router / runtime 的外层组织
- 可见工具集整理
- hidden tools 过滤

### 5.2 LangGraph tool loop runtime

- `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/retrieval/langgraph_runtime.py`

当前默认 runtime 机制的关键点：

- `max_tool_rounds_per_run` 默认是 `3`
- 第一轮最多可调用 `5` 个互补工具
- 后续每轮最多 `1` 个工具
- 第一轮工具支持并行执行
- 后续轮次走串行补证据

这是最近几轮对 runtime 的主要锚点之一。

### 5.3 知识抽取相关

- narrative graph：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/builder/narrative_graph_builder.py`
- interaction extraction：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/interaction_extraction_agent.py`
- property extraction：
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/core/agent/property_extraction_agent.py`

这些文件最近有未提交改动，主要和知识图谱构建加速方向有关。


## 6. 本地 dirty state

当前工作区不是干净的。`git diff --stat` 显示，已修改但未提交的核心文件包括：

- `core/agent/interaction_extraction_agent.py`
- `core/agent/property_extraction_agent.py`
- `core/agent/retrieval/langgraph_runtime.py`
- `core/agent/retriever_agent_langchain.py`
- `core/builder/narrative_graph_builder.py`
- `experiments/quality/run_online_setting_from_workspace.py`
- `experiments/quality/run_quality_benchmark.py`

另外有大量未跟踪文件，尤其是：

- `experiments/fiarytableqa/`
- `experiments/quality/artifacts/regression/*`
- `experiments/quality/assets/`

因此下一次接手前，不能假设 repo 是干净的，也不能贸然做大范围回滚。


## 7. 尚未完成但已经明确的事项

### 7.1 FiarytableQA 当前 run 结束后要做的事

1. 检查：
   - `summary.json`
   - `summary.md`
2. 确认 `type_breakdown` 是否正确生成。
3. 汇总最终：
   - article count
   - question count
   - overall / pass / avg_judge / avg_latency
   - 各 question type 的结果

如果 `summary` 没有正确写出 `type_breakdown`，优先做离线 summary 重建，不必重跑整批实验。

### 7.2 知识图谱构建提速

用户已经明确指定了只考虑以下三个优化方向，不要扩散修改范围：

1. `narrative graph phase1`
   - 并行抽局部 episodes，再做 doc-level merge
   - 对很短的 part 做 pack，减少 update 轮数
2. 对短 `full_text` 节点直接 single-shot
   - 不走 chunk+merge 双阶段
3. `interaction extraction`
   - occasion 内分段并行抽取
   - 最后统一 normalize + dedupe

限制条件也很明确：

- 不要碰会显著改变图质量的地方
- 以加速为主，不做大改范式


## 8. benchmark 规模信息

当前已确认的数据集规模：

- `FiarytableQA`
  - `278` 篇文章
  - `10580` 道题
- `QUALITY`
  - `162` 篇文章
  - `2523` 道题


## 9. 快速恢复命令

### 9.1 查看 FiarytableQA 当前进度

```bash
python3 - <<'PY'
from pathlib import Path
import json
reports=Path('/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/reports')
files=sorted([p for p in reports.glob('*.json') if p.name not in {'progress.json','summary.json'}])
print('article_reports', len(files))
progress=reports/'progress.json'
if progress.exists():
    print(progress.read_text(encoding='utf-8'))
PY
```

### 9.2 查看 tmux 日志尾部

```bash
source /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/activate screenplay
tmux capture-pane -pt fiarytableqa100_20260412:0 | tail -n 80
```

### 9.3 计算当前部分指标

```bash
python3 - <<'PY'
from pathlib import Path
import json
reports=Path('/vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver_langgraph/experiments/fiarytableqa/runs/fiarytableqa_no_strategy_100_20260412/reports')
files=sorted([p for p in reports.glob('*.json') if p.name not in {'progress.json','summary.json'}])
all_results=[]; qcount=0
for p in files:
    data=json.loads(p.read_text(encoding='utf-8'))
    qcount += int(data.get('question_count',0) or 0)
    all_results.extend(data.get('results') or [])
print('completed_articles', len(files))
print('completed_questions', qcount)
if all_results:
    overall=sum(1 for r in all_results if r.get('is_correct'))/len(all_results)
    grouped={}
    for r in all_results:
        grouped.setdefault((r.get('article_name'), r.get('question_id')), []).append(bool(r.get('is_correct')))
    pass_acc=sum(1 for vals in grouped.values() if any(vals))/len(grouped)
    scores=[float(((r.get('judge') or {}).get('score',0.0) or 0.0)) for r in all_results]
    lat=[int(r.get('latency_ms',0) or 0) for r in all_results]
    print(round(overall,4), round(pass_acc,4), round(sum(scores)/len(scores),4), round(sum(lat)/len(lat),2))
PY
```


## 10. 接下来最合理的顺序

1. 等当前 `FiarytableQA 100` 篇 run 结束。
2. 验证 `summary.json` / `summary.md` / `type_breakdown`。
3. 固化最终结果。
4. 再处理知识图谱构建提速，严格限制在用户指定的三个点上。

不要在当前 run 尚未结束时同时做大范围 runtime / router 变更，否则实验语义会再次混杂。
