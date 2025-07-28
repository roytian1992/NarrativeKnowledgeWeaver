import pickle
import networkx as nx
from pyvis.network import Network
import gradio as gr
import tempfile
import base64
import os
import re

# 加载图数据
with open("data/event_causality_graph/event_causality_graph.pickle", "rb") as f:
    G_full = pickle.load(f)

# 自动换行工具函数
def wrap_text(text, width: int = 20) -> str:
    """
    将输入文本按指定宽度换行。
    
    Args:
        text (str): 待处理文本。
        width (int): 每行最多字符数（默认为 10）。
    
    Returns:
        str: 处理后的多行文本。
    """
    # 确保是字符串
    text = str(text)

    # 逐段落处理，保留原本的换行
    wrapped_lines = []
    for paragraph in text.splitlines():
        # 按 width 切分当前段落
        wrapped_lines.extend(
            paragraph[i : i + width] for i in range(0, len(paragraph), width)
        )

    return "\n".join(wrapped_lines)


# 如果希望按标点断行：
# def wrap_text(text): return re.sub(r"(。|，|；)", r"\1<br>", text)

# 主函数：构建并展示图
def visualize_graph_embed(max_nodes=30, levels=["High", "Medium", "Low"]):
    # 1. 筛选边和子图
    selected_edges = [
        (u, v, d) for u, v, d in G_full.edges(data=True)
        if d.get("causal_level") in levels
    ]
    subG = nx.DiGraph()
    for u, v, d in selected_edges:
        subG.add_edge(u, v, **d)
        for n in [u, v]:
            if n in G_full.nodes:
                subG.add_node(n, **G_full.nodes[n])

    subG = subG.subgraph(list(subG.nodes)[:max_nodes])

    # 2. 构建 pyvis 网络
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(subG)

    # 3. 设置节点 tooltip 与 label
    for node in net.nodes:
        nid = node['id']
        data = G_full.nodes[nid]
        name = data.get("name", "")
        desc = data.get("description", "")
        scene = data.get("scene_name", "N/A")
        node['label'] = name
        node['title'] = f"{name}\nScene: {scene}\n{desc}"
        node['color'] = "#97C2FC"

    # 4. 设置边 tooltip 与颜色
    color_map = {"High": "crimson", "Medium": "orange", "Low": "gray"}
    for edge in net.edges:
        u, v = edge['from'], edge['to']
        d = G_full.edges.get((u, v), {})
        level = d.get("causal_level", "Low")
        reason = wrap_text(d.get("reason", ""))
        edge['color'] = color_map.get(level, "gray")
        edge['title'] = f"{level} 因果\n{wrap_text(reason)}"

    # 5. 写入 HTML 文件并转 base64
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        with open(tmp_file.name, "rb") as f:
            html_data = f.read()
        encoded = base64.b64encode(html_data).decode("utf-8")
        data_url = f"data:text/html;base64,{encoded}"
        os.remove(tmp_file.name)

    # 6. 生成嵌入 iframe
    iframe = f'<iframe src="{data_url}" width="100%" height="640px" frameborder="0"></iframe>'
    return iframe


# Gradio UI
demo = gr.Interface(
    fn=visualize_graph_embed,
    inputs=[
        gr.Slider(minimum=10, maximum=300, value=40, step=5, label="最多展示节点数量"),
        gr.CheckboxGroup(["High", "Medium", "Low"], value=["High", "Medium"], label="显示因果强度等级")
    ],
    outputs=gr.HTML(label="交互式事件因果图"),
    title="📌 事件因果图交互可视化",
    description="选择节点数量与因果等级，查看含 scene_name 与摘要描述的事件图"
)

if __name__ == "__main__":
    demo.launch()
