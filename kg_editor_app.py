"""
知识图谱编辑器 - Gradio 应用
提供实体查询、实体编辑和关系编辑功能
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
import base64

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None


class KnowledgeGraphEditor:
    """知识图谱编辑器核心类"""
    
    def __init__(self, graph_query_utils):
        """
        初始化编辑器
        
        Args:
            graph_query_utils: GraphQueryUtils 实例
        """
        self.graph_query_utils = graph_query_utils

    def _graph(self):
        return self.graph_query_utils.graph_store.get_graph()

    def _persist(self) -> None:
        self.graph_query_utils.graph_store.persist()

    def _node_labels(self, node_data: Dict[str, Any]) -> List[str]:
        if hasattr(self.graph_query_utils, "_node_labels"):
            labels = list(self.graph_query_utils._node_labels(node_data))
        else:
            labels = list(node_data.get("type", []) or [])
        return [label for label in labels if label and label != "Entity"]

    @staticmethod
    def _relation_type(rel_data: Dict[str, Any]) -> str:
        return (
            str(rel_data.get("predicate") or rel_data.get("relation_type") or rel_data.get("type") or "RELATED_TO")
            .strip()
            or "RELATED_TO"
        )

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _remove_flattened_props(target: Dict[str, Any], props: Dict[str, Any], reserved: set[str]) -> None:
        for key in props.keys():
            if key in reserved:
                continue
            target.pop(key, None)
    
    # ==================== 实体查询功能 ====================
    
    def search_entities(self, keyword: str = "", entity_type: str = "") -> Tuple[str, str]:
        """
        搜索实体
        
        Args:
            keyword: 搜索关键词
            entity_type: 实体类型（可选）
            
        Returns:
            (搜索结果表格HTML, 状态信息)
        """
        try:
            wanted_type = str(entity_type or "").strip()
            if wanted_type == "全部":
                wanted_type = ""
            if not keyword and not wanted_type:
                return "<p style='color: orange;'>请输入搜索关键词或选择实体类型</p>", "⚠️ 请提供搜索条件"

            graph = self._graph()
            kw = str(keyword or "").strip().lower()

            results = []
            for node_id, data in graph.nodes(data=True):
                labels = self._node_labels(data)
                if wanted_type and wanted_type not in labels:
                    continue

                name = str(data.get("name", "") or "")
                aliases = [str(alias or "") for alias in data.get("aliases", []) or []]
                if kw:
                    haystacks = [name.lower()] + [alias.lower() for alias in aliases]
                    if not any(kw in item for item in haystacks):
                        continue

                results.append(
                    {
                        "id": str(data.get("id") or node_id),
                        "name": name or str(node_id),
                        "types": labels or ["Entity"],
                        "description": str(data.get("description", "") or ""),
                    }
                )

            results.sort(key=lambda row: (row["name"], row["id"]))
            results = results[:50]

            if not results:
                return "<p style='color: gray;'>未找到匹配的实体</p>", "ℹ️ 未找到结果"
            
            # 构造结果表格（带点击跳转功能）
            html = """
            <style>
                .entity-row {
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                .entity-row:hover {
                    background-color: #e8f4f8 !important;
                }
            </style>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background-color: #f0f0f0;'>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>实体ID</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>名称</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>类型</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>描述</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for record in results:
                entity_id = record.get('id', 'N/A')
                name = record.get('name', 'N/A')
                types = ', '.join(record.get('types', []))
                description = record.get('description', 'N/A')
                
                # 截断过长的描述
                if len(description) > 100:
                    description = description[:100] + "..."
                
                # 添加点击事件，使用 data-entity-id 属性
                html += f"""
                    <tr class='entity-row' onclick='
                        const entityId = "{entity_id}";
                        const tabs = document.querySelectorAll("button[id*=\\"-tab-\\"");
                        const editTab = Array.from(tabs).find(t => t.textContent.includes("实体编辑"));
                        if (editTab) editTab.click();
                        setTimeout(() => {{
                            const entityIdInput = document.querySelector("textarea[placeholder*=\\"ent_\\"]");
                            if (entityIdInput) {{
                                entityIdInput.value = entityId;
                                entityIdInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                                const loadBtn = Array.from(document.querySelectorAll("button")).find(b => b.textContent.includes("加载实体"));
                                if (loadBtn) loadBtn.click();
                            }}
                        }}, 300);
                    ' style='cursor: pointer;'>
                        <td style='border: 1px solid #ddd; padding: 8px;'><code>{entity_id}</code></td>
                        <td style='border: 1px solid #ddd; padding: 8px;'><strong>{name}</strong></td>
                        <td style='border: 1px solid #ddd; padding: 8px;'>{types}</td>
                        <td style='border: 1px solid #ddd; padding: 8px;'>{description}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            <p style='color: #666; font-size: 12px; margin-top: 10px;'>💡 点击任意行可跳转到实体编辑页面</p>
            """
            
            status = f"✅ 找到 {len(results)} 个实体"
            return html, status
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>查询出错: {str(e)}</p>"
            return error_msg, f"❌ 错误: {str(e)}"
    
    def get_entity_types(self) -> List[str]:
        """获取所有实体类型"""
        try:
            types = [label for label in self.graph_query_utils.list_entity_types() if label and label != "Entity"]
            return ["全部"] + sorted(types)
        except Exception as e:
            print(f"获取实体类型出错: {e}")
            return ["全部"]
    
    # ==================== 实体编辑功能 ====================
    
    def load_entity(self, entity_id: str) -> Tuple[str, str, str, str, str, str, str]:
        """
        根据实体ID加载实体信息
        
        Args:
            entity_id: 实体ID
            
        Returns:
            (实体名称, 实体类型, 描述, 别名JSON, 属性JSON, embedding状态, 状态信息)
        """
        try:
            if not entity_id or not entity_id.strip():
                return "", "", "", "", "", "", "⚠️ 请输入实体ID"

            node_id = entity_id.strip()
            graph = self._graph()
            if not graph.has_node(node_id):
                return "", "", "", "", "", "", f"❌ 未找到实体: {entity_id}"

            entity_node = dict(graph.nodes[node_id])

            # 提取实体信息
            entity_name = entity_node.get('name', '')
            entity_description = entity_node.get('description', '')

            entity_types = ', '.join(self._node_labels(entity_node)) or 'Unknown'

            # 提取别名
            aliases = entity_node.get('aliases', [])
            aliases_json = json.dumps(aliases, ensure_ascii=False, indent=2)
            
            # 检查 embedding 状态
            embedding = entity_node.get('embedding', None)
            if embedding and len(embedding) > 0:
                embedding_status = f"✅ 已计算 (维度: {len(embedding)})"
            else:
                embedding_status = "❌ 未计算"
            
            # 提取 properties（如果是嵌套的字典）
            properties = entity_node.get('properties', {})
            
            # 如果 properties 是字符串，尝试解析
            if isinstance(properties, str):
                try:
                    properties = json.loads(properties)
                except:
                    properties = {}
            
            # 过滤掉不需要展示的字段
            excluded_fields = ['id', 'name', 'description', 'aliases', 'embedding', 'source_chunks', 
                             'additional_chunks', 'scope', 'version', 'types', 'type']
            
            # 如果 properties 为空，从节点本身提取
            if not properties:
                if hasattr(entity_node, 'items'):
                    properties = {k: v for k, v in entity_node.items() 
                                if k not in excluded_fields and not k.startswith('_')}
                else:
                    properties = {}
            
            properties_json = json.dumps(properties, ensure_ascii=False, indent=2)
            
            status = f"✅ 已加载实体: {entity_name}"
            return entity_name, entity_types, entity_description, aliases_json, properties_json, embedding_status, status
            
        except Exception as e:
            return "", "", "", "", "", "", f"❌ 加载出错: {str(e)}"
    
    def _generate_relations_list(self, center_id: str, center_name: str, 
                                 outgoing: List[Dict], incoming: List[Dict]) -> str:
        """
        生成关系列表 HTML

        - 顶部展示中心节点及一个“复制中心ID”的按钮
        - 表格中增加一列“相关实体ID”
        - 每行有一个“复制ID”按钮，将该行相关实体的 id 复制到剪贴板
        """

        def html_escape(s: str) -> str:
            if s is None:
                s = ""
            s = str(s)
            return (
                s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;")
                 .replace("'", "&#39;")
            )

        def js_str_escape(s: str) -> str:
            """用于放进 onclick 里的 JS 字符串（单引号包裹）"""
            if s is None:
                s = ""
            s = str(s)
            return (
                s.replace("\\", "\\\\")
                 .replace("'", "\\'")
            )

        html = f"""
        <div style='margin-bottom: 8px;'>
            <strong>中心节点:</strong> {center_name} ({center_id})
            <button
                style="margin-left: 8px; padding: 2px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc;"
                onclick="navigator.clipboard.writeText('{js_str_escape(center_id)}').then(() => {{
                    alert('中心节点ID已复制到剪贴板：{js_str_escape(center_id)}');
                }}).catch(err => {{
                    alert('复制失败，请手动复制：{js_str_escape(center_id)}');
                }});"
            >
                复制中心ID
            </button>
        </div>
        <p style='color:#666;font-size:13px;margin-bottom:6px;'>
            💡 提示：点击“复制ID”按钮，将相关实体的ID复制到剪贴板；然后你可以在「关系编辑」里手动粘贴到“目标实体ID”。
        </p>
        <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
            <thead>
                <tr style='background-color: #f0f0f0;'>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>方向</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系类型</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>相关实体</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>相关实体ID</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left; width: 90px;'>操作</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>属性预览</th>
                </tr>
            </thead>
            <tbody>
        """

        def get_properties_preview(rel_obj) -> str:
            """只展示关系里的 properties 字段"""
            rel_props = dict(rel_obj) if hasattr(rel_obj, "__iter__") else {}
            value = rel_props.get("properties", "")
            if value is None:
                value = ""
            # 如果是 dict/list 就转成 json，否则直接转字符串
            try:
                if isinstance(value, (dict, list)):
                    preview = json.dumps(value, ensure_ascii=False)
                else:
                    preview = str(value)
            except Exception:
                preview = str(value)
            # 适当截断
            return preview[:200]

        # ========= 出边：center -> neighbor =========
        for item in outgoing:
            neighbor = item.get("node")
            rel = item.get("rel")
            if neighbor is None or rel is None:
                continue

            neighbor_id = neighbor.get("id", "unknown")
            neighbor_name = neighbor.get("name", "Unknown")
            rel_type = self._relation_type(rel or {})

            preview = get_properties_preview(rel)

            html += f"""
                <tr class='relation-row'>
                    <td style='border: 1px solid #ddd; padding: 8px;'>→ 出边</td>
                    <td style='border: 1px solid #ddd; padding: 8px;'><strong>{html_escape(rel_type)}</strong></td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>{html_escape(neighbor_name)}</td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>
                        <code>{html_escape(neighbor_id)}</code>
                    </td>
                    <td style='border: 1px solid #ddd; padding: 8px; text-align:center;'>
                        <button
                            style="padding: 2px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc;"
                            onclick="navigator.clipboard.writeText('{js_str_escape(neighbor_id)}').then(() => {{
                                alert('相关实体ID已复制：{js_str_escape(neighbor_id)}');
                            }}).catch(err => {{
                                alert('复制失败，请手动复制：{js_str_escape(neighbor_id)}');
                            }});"
                        >
                            复制ID
                        </button>
                    </td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>
                        <pre style='margin: 0; font-size: 12px;'>{html_escape(preview)}</pre>
                    </td>
                </tr>
            """

        # ========= 入边：other -> center =========
        for item in incoming:
            other = item.get("node")
            rel = item.get("rel")
            if other is None or rel is None:
                continue

            other_id = other.get("id", "unknown")
            other_name = other.get("name", "Unknown")
            rel_type = self._relation_type(rel or {})

            preview = get_properties_preview(rel)

            html += f"""
                <tr class='relation-row'>
                    <td style='border: 1px solid #ddd; padding: 8px;'>← 入边</td>
                    <td style='border: 1px solid #ddd; padding: 8px;'><strong>{html_escape(rel_type)}</strong></td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>{html_escape(other_name)}</td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>
                        <code>{html_escape(other_id)}</code>
                    </td>
                    <td style='border: 1px solid #ddd; padding: 8px; text-align:center;'>
                        <button
                            style="padding: 2px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc;"
                            onclick="navigator.clipboard.writeText('{js_str_escape(other_id)}').then(() => {{
                                alert('相关实体ID已复制：{js_str_escape(other_id)}');
                            }}).catch(err => {{
                                alert('复制失败，请手动复制：{js_str_escape(other_id)}');
                            }});"
                        >
                            复制ID
                        </button>
                    </td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>
                        <pre style='margin: 0; font-size: 12px;'>{html_escape(preview)}</pre>
                    </td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html


    def visualize_ego_network(self, entity_id: str) -> Tuple[str, str, str]:
        """
        可视化实体的 1-hop ego network

        Args:
            entity_id: 实体ID

        Returns:
            (关系列表 HTML, 可视化 HTML(iframe), 状态信息)
        """
        try:
            if Network is None:
                return "", "", "❌ 未安装 pyvis，无法生成可视化"
            if not entity_id or not entity_id.strip():
                # 注意：这里要返回 3 个值，对应 gradio 的 3 个输出
                return "", "", "⚠️ 请输入实体ID"

            graph = self._graph()
            center_id = entity_id.strip()
            if not graph.has_node(center_id):
                return "", "", f"❌ 未找到实体: {entity_id}"

            center_node = dict(graph.nodes[center_id])
            center_node.setdefault("id", center_id)
            outgoing = []
            incoming = []
            for _src, dst, _key, data in graph.out_edges(center_id, keys=True, data=True):
                if not graph.has_node(dst):
                    continue
                outgoing.append({"node": dict(graph.nodes[dst]), "rel": dict(data)})
            for src, _dst, _key, data in graph.in_edges(center_id, keys=True, data=True):
                if not graph.has_node(src):
                    continue
                incoming.append({"node": dict(graph.nodes[src]), "rel": dict(data)})

            # 创建网络图，使用 CDN 资源
            net = Network(
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#000000",
                cdn_resources="remote",
            )
            net.set_options(
                """
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.04
                    },
                    "stabilization": {
                        "iterations": 200
                    }
                },
                "nodes": {
                    "font": {
                        "size": 14
                    }
                },
                "edges": {
                    "font": {
                        "size": 12,
                        "align": "middle"
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                        }
                    },
                    "smooth": {
                        "type": "continuous"
                    }
                }
            }
            """
            )

            # 添加中心节点
            center_id = center_node.get("id", entity_id)
            center_name = center_node.get("name", "Unknown")
            center_type = ", ".join(self._node_labels(center_node)) or "Unknown"

            net.add_node(
                center_id,
                label=center_name,
                title=f"ID: {center_id}\nType: {center_type}",
                color="#ff6b6b",
                size=30,
                font={"size": 16, "bold": True},
            )

            # 添加出边邻居
            for item in outgoing:
                neighbor = item.get("node")
                rel = item.get("rel")

                if neighbor is None:
                    continue

                neighbor_id = neighbor.get("id", "unknown")
                neighbor_name = neighbor.get("name", "Unknown")
                neighbor_type = ", ".join(self._node_labels(neighbor)) or "Unknown"

                net.add_node(
                    neighbor_id,
                    label=neighbor_name,
                    title=f"ID: {neighbor_id}\nType: {neighbor_type}",
                    color="#4ecdc4",
                    size=20,
                )

                if rel is not None:
                    rel_type = self._relation_type(rel)
                    net.add_edge(
                        center_id,
                        neighbor_id,
                        label=rel_type,
                        title=rel_type,
                        color="#999999",
                    )

            # 添加入边邻居
            for item in incoming:
                other = item.get("node")
                rel = item.get("rel")

                if other is None:
                    continue

                other_id = other.get("id", "unknown")
                other_name = other.get("name", "Unknown")
                other_type = ", ".join(self._node_labels(other)) or "Unknown"

                if other_id not in [node["id"] for node in net.nodes]:
                    net.add_node(
                        other_id,
                        label=other_name,
                        title=f"ID: {other_id}\nType: {other_type}",
                        color="#95e1d3",
                        size=20,
                    )

                if rel is not None:
                    rel_type = self._relation_type(rel)
                    net.add_edge(
                        other_id,
                        center_id,
                        label=rel_type,
                        title=rel_type,
                        color="#999999",
                    )

            # 生成关系列表 HTML
            relations_html = self._generate_relations_list(
                center_id, center_name, outgoing, incoming
            )

            total_neighbors = len(
                [n for n in outgoing if n.get("node") is not None]
            ) + len([n for n in incoming if n.get("node") is not None])

            status = f"✅ 已生成 Ego Network 可视化 (中心节点 + {total_neighbors} 个邻居)"

            # —— 关键：像旧版一样写临时文件 + base64 + iframe —— #
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".html", mode="w", encoding="utf-8"
            ) as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, "rb") as f:
                html_data = f.read()
            encoded = base64.b64encode(html_data).decode("utf-8")
            data_url = f"data:text/html;base64,{encoded}"
            os.remove(tmp_file_path)

            iframe_html = (
                f'<iframe src="{data_url}" '
                f'width="100%" height="650px" frameborder="0"></iframe>'
            )

            return relations_html, iframe_html, status

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"可视化出错: {error_detail}")
            return "", "", f"❌ 可视化出错: {str(e)}"

    def update_entity(self, entity_id: str, entity_name: str, entity_description: str,
                     aliases_json: str, properties_json: str) -> str:
        """
        更新实体信息
        
        Args:
            entity_id: 实体ID
            entity_name: 实体名称
            entity_description: 实体描述
            aliases_json: 别名JSON字符串
            properties_json: 属性JSON字符串
            
        Returns:
            状态信息
        """
        try:
            if not entity_id or not entity_id.strip():
                return "⚠️ 请输入实体ID"

            # 解析别名JSON
            try:
                aliases = json.loads(aliases_json) if aliases_json else []
            except json.JSONDecodeError as e:
                return f"❌ 别名JSON格式错误: {str(e)}"
            if not isinstance(aliases, list):
                return "❌ 别名JSON必须是数组"

            # 解析属性JSON
            try:
                properties = json.loads(properties_json) if properties_json else {}
            except json.JSONDecodeError as e:
                return f"❌ 属性JSON格式错误: {str(e)}"
            if not isinstance(properties, dict):
                return "❌ 属性JSON必须是对象"

            node_id = entity_id.strip()
            graph = self._graph()
            if not graph.has_node(node_id):
                return f"❌ 未找到实体: {entity_id}"

            entity_node = graph.nodes[node_id]
            reserved_fields = {
                "id", "name", "description", "aliases", "embedding", "source_chunks",
                "additional_chunks", "scope", "version", "types", "type", "labels",
                "source_documents", "community_id", "community_path", "pagerank",
                "degree", "betweenness",
            }
            old_properties = self._safe_dict(entity_node.get("properties", {}))
            self._remove_flattened_props(entity_node, old_properties, reserved_fields)

            entity_node["name"] = entity_name
            entity_node["description"] = entity_description
            entity_node["aliases"] = aliases
            entity_node["properties"] = properties
            for key, value in properties.items():
                if key in reserved_fields:
                    continue
                entity_node[key] = value

            self._persist()
            return f"✅ 实体更新成功: {entity_id}"
            
        except Exception as e:
            return f"❌ 更新出错: {str(e)}"
    
    def delete_entity(self, entity_id: str) -> str:
        """
        删除实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            状态信息
        """
        try:
            if not entity_id or not entity_id.strip():
                return "⚠️ 请输入实体ID"

            node_id = entity_id.strip()
            graph = self._graph()
            if not graph.has_node(node_id):
                return f"❌ 未找到实体: {entity_id}"

            entity_name = graph.nodes[node_id].get('name', entity_id)
            graph.remove_node(node_id)
            self._persist()
            return f"✅ 实体已删除: {entity_name} ({entity_id})"
            
        except Exception as e:
            return f"❌ 删除出错: {str(e)}"
    
    def recalculate_embedding(self, entity_id: str) -> Tuple[str, str]:
        """
        重新计算实体的 embedding
        
        Args:
            entity_id: 实体ID
            
        Returns:
            (embedding状态, 状态信息)
        """
        try:
            if not entity_id or not entity_id.strip():
                return "", "⚠️ 请输入实体ID"

            node_id = entity_id.strip()
            graph = self._graph()
            if not graph.has_node(node_id):
                return "", f"❌ 未找到实体: {entity_id}"

            entity_node = dict(graph.nodes[node_id])
            # 提取实体类型
            entity_types = self._node_labels(entity_node)
            node_type = entity_types[0] if entity_types else ""

            # 构造节点数据用于编码
            node_data = {
                "name": entity_node.get('name', ''),
                "description": entity_node.get('description', ''),
                "type": node_type,
                "properties": entity_node.get('properties', {})
            }
            
            # 如果 properties 是字符串，尝试解析
            if isinstance(node_data["properties"], str):
                try:
                    node_data["properties"] = json.loads(node_data["properties"])
                except:
                    node_data["properties"] = {}
            
            # 调用 graph_query_utils 的 encode_node_embedding 方法
            if not hasattr(self.graph_query_utils, 'encode_node_embedding'):
                return "", "❌ 当前图后端不支持 encode_node_embedding 方法"
            
            embedding = self.graph_query_utils.encode_node_embedding(node_data)

            self.graph_query_utils.update_node_embedding(node_id, embedding)
            
            embedding_status = f"✅ 已计算 (维度: {len(embedding)})"
            return embedding_status, f"✅ Embedding 重新计算成功: {entity_id}"
            
        except Exception as e:
            return "", f"❌ 重新计算 Embedding 出错: {str(e)}"
    
    # ==================== 关系编辑功能 ====================
    
    def load_relations(self, src_entity_id: str, tgt_entity_id: str) -> Tuple[str, str]:
        """
        加载两个实体之间的关系
        
        Args:
            src_entity_id: 源实体ID
            tgt_entity_id: 目标实体ID
            
        Returns:
            (关系列表HTML, 状态信息)
        """
        try:
            if not src_entity_id or not tgt_entity_id:
                return "<p style='color: orange;'>请输入源实体ID和目标实体ID</p>", "⚠️ 请输入完整的实体ID", "", "{}"

            src_id = src_entity_id.strip()
            tgt_id = tgt_entity_id.strip()
            graph = self._graph()
            results = []

            if graph.has_node(src_id) and graph.has_node(tgt_id):
                for _src, _dst, _key, data in graph.out_edges(src_id, keys=True, data=True):
                    if _dst != tgt_id:
                        continue
                    results.append(
                        {
                            "src_name": graph.nodes[src_id].get("name", src_id),
                            "src_id": src_id,
                            "rel_type": self._relation_type(data),
                            "rel_props": dict(data),
                            "tgt_name": graph.nodes[tgt_id].get("name", tgt_id),
                            "tgt_id": tgt_id,
                            "direction": "forward",
                        }
                    )
                for _src, _dst, _key, data in graph.out_edges(tgt_id, keys=True, data=True):
                    if _dst != src_id:
                        continue
                    results.append(
                        {
                            "src_name": graph.nodes[tgt_id].get("name", tgt_id),
                            "src_id": tgt_id,
                            "rel_type": self._relation_type(data),
                            "rel_props": dict(data),
                            "tgt_name": graph.nodes[src_id].get("name", src_id),
                            "tgt_id": src_id,
                            "direction": "reverse",
                        }
                    )

            if not results:
                return "<p style='color: gray;'>未找到这两个实体之间的关系</p>", "ℹ️ 未找到关系", "", "{}"
            
            # 构造结果表格
            html = f"""
            <div style='margin-bottom: 10px;'>
                <strong>输入的实体:</strong> {src_entity_id} 和 {tgt_entity_id}
            </div>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background-color: #f0f0f0;'>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系方向</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系类型</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系属性</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for record in results:
                rel_type = record.get('rel_type', 'N/A')
                rel_props = record.get('rel_props', {})
                direction = record.get('direction', 'forward')
                src_name = record.get('src_name', 'N/A')
                tgt_name = record.get('tgt_name', 'N/A')
                actual_src_id = record.get('src_id', 'N/A')
                actual_tgt_id = record.get('tgt_id', 'N/A')
                
                # 显示关系方向
                if direction == 'forward':
                    direction_str = f"{src_name} → {tgt_name}"
                else:
                    direction_str = f"{src_name} → {tgt_name} (反向)"
                
                # 如果 rel_props 中有 properties 字段，解析它
                if 'properties' in rel_props:
                    properties_str = rel_props['properties']
                    if isinstance(properties_str, str):
                        try:
                            properties_obj = json.loads(properties_str)
                            rel_props_str = json.dumps(properties_obj, ensure_ascii=False, indent=2)
                        except:
                            rel_props_str = properties_str
                    else:
                        rel_props_str = json.dumps(properties_str, ensure_ascii=False, indent=2)
                else:
                    rel_props_str = json.dumps(rel_props, ensure_ascii=False, indent=2) if rel_props else "{}"
                
                html += f"""
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 8px;'>{direction_str}<br><small style='color: #666;'>{actual_src_id} → {actual_tgt_id}</small></td>
                        <td style='border: 1px solid #ddd; padding: 8px;'><strong>{rel_type}</strong></td>
                        <td style='border: 1px solid #ddd; padding: 8px;'><pre style='margin: 0;'>{rel_props_str}</pre></td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
            
            status = f"✅ 找到 {len(results)} 个关系"
            
            # 返回第一个关系的类型和属性，用于自动填充
            first_rel_type = results[0].get('rel_type', '')
            first_rel_props = results[0].get('rel_props', {})
            
            # 解析 properties 字段
            if 'properties' in first_rel_props:
                properties_str = first_rel_props['properties']
                if isinstance(properties_str, str):
                    try:
                        properties_obj = json.loads(properties_str)
                        first_rel_props_json = json.dumps(properties_obj, ensure_ascii=False, indent=2)
                    except:
                        first_rel_props_json = properties_str
                else:
                    first_rel_props_json = json.dumps(properties_str, ensure_ascii=False, indent=2)
            else:
                first_rel_props_json = json.dumps(first_rel_props, ensure_ascii=False, indent=2) if first_rel_props else "{}"
            
            return html, status, first_rel_type, first_rel_props_json
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>查询出错: {str(e)}</p>"
            return error_msg, f"❌ 错误: {str(e)}", "", "{}"
    
    def update_relation(self, src_entity_id: str, tgt_entity_id: str, 
                       rel_type: str, new_rel_type: str, properties_json: str) -> str:
        """
        更新关系属性，如果提供了新关系类型则修改关系类型
        
        Args:
            src_entity_id: 源实体ID
            tgt_entity_id: 目标实体ID
            rel_type: 当前关系类型
            new_rel_type: 新关系类型（如果不修改则为空）
            properties_json: 关系属性JSON字符串
            
        Returns:
            状态信息
        """
        try:
            if not src_entity_id or not tgt_entity_id or not rel_type:
                return "⚠️ 请输入完整的实体ID和关系类型"

            # 解析属性JSON
            try:
                properties = json.loads(properties_json)
            except json.JSONDecodeError as e:
                return f"❌ 属性JSON格式错误: {str(e)}"
            if not isinstance(properties, dict):
                return "❌ 属性JSON必须是对象"

            src_id = src_entity_id.strip()
            tgt_id = tgt_entity_id.strip()
            current_type = rel_type.strip()
            target_type = (new_rel_type or "").strip() or current_type
            graph = self._graph()
            if not graph.has_node(src_id) or not graph.has_node(tgt_id):
                return f"❌ 未找到指定的关系: {rel_type}"

            matched = []
            for _src, _dst, key, data in graph.out_edges(src_id, keys=True, data=True):
                if _dst != tgt_id or self._relation_type(data) != current_type:
                    continue
                matched.append((key, data))

            if not matched:
                return f"❌ 未找到指定的关系: {rel_type}"

            reserved_fields = {
                "id", "subject_id", "object_id", "predicate", "relation_type", "type",
                "relation_name", "persistence", "description", "confidence",
                "properties", "source_documents", "embedding",
            }
            for key, rel_data in matched:
                old_properties = self._safe_dict(rel_data.get("properties", {}))
                self._remove_flattened_props(rel_data, old_properties, reserved_fields)
                rel_data["id"] = rel_data.get("id") or key
                rel_data["subject_id"] = src_id
                rel_data["object_id"] = tgt_id
                rel_data["predicate"] = target_type
                rel_data["relation_type"] = target_type
                rel_data["properties"] = properties
                for prop_key, prop_value in properties.items():
                    if prop_key in reserved_fields:
                        continue
                    rel_data[prop_key] = prop_value

            self._persist()
            if target_type != current_type:
                return f"✅ 关系类型已修改: {src_entity_id} -[{current_type}]-> {tgt_entity_id} → {src_entity_id} -[{target_type}]-> {tgt_entity_id}"
            return f"✅ 关系更新成功: {src_entity_id} -[{current_type}]-> {tgt_entity_id}"
            
        except Exception as e:
            return f"❌ 更新出错: {str(e)}"
    
    def delete_relation(self, src_entity_id: str, tgt_entity_id: str, rel_type: str) -> str:
        """
        删除关系
        
        Args:
            src_entity_id: 源实体ID
            tgt_entity_id: 目标实体ID
            rel_type: 关系类型
            
        Returns:
            状态信息
        """
        try:
            if not src_entity_id or not tgt_entity_id or not rel_type:
                return "⚠️ 请输入完整的实体ID和关系类型"

            src_id = src_entity_id.strip()
            tgt_id = tgt_entity_id.strip()
            target_type = rel_type.strip()
            graph = self._graph()
            if not graph.has_node(src_id) or not graph.has_node(tgt_id):
                return f"❌ 未找到指定的关系: {rel_type}"

            matched_keys = []
            for _src, _dst, key, data in graph.out_edges(src_id, keys=True, data=True):
                if _dst != tgt_id or self._relation_type(data) != target_type:
                    continue
                matched_keys.append(key)

            if not matched_keys:
                return f"❌ 未找到指定的关系: {rel_type}"

            src_name = graph.nodes[src_id].get('name', src_entity_id)
            tgt_name = graph.nodes[tgt_id].get('name', tgt_entity_id)
            for key in matched_keys:
                graph.remove_edge(src_id, tgt_id, key)
            self._persist()
            return f"✅ 关系已删除: {src_name} -[{rel_type}]-> {tgt_name}"
            
        except Exception as e:
            return f"❌ 删除出错: {str(e)}"
    
    def get_relation_types(self, src_entity_id: str, tgt_entity_id: str) -> List[str]:
        """获取两个实体之间的关系类型"""
        try:
            if not src_entity_id or not tgt_entity_id:
                return [""]

            src_id = src_entity_id.strip()
            tgt_id = tgt_entity_id.strip()
            graph = self._graph()
            if not graph.has_node(src_id) or not graph.has_node(tgt_id):
                return [""]

            relation_types = []
            for _src, _dst, _key, data in graph.out_edges(src_id, keys=True, data=True):
                if _dst != tgt_id:
                    continue
                rel_name = self._relation_type(data)
                if rel_name not in relation_types:
                    relation_types.append(rel_name)
            return relation_types or [""]
        except Exception as e:
            print(f"获取关系类型出错: {e}")
            return [""]


def create_gradio_interface(graph_query_utils):
    """
    创建Gradio界面
    
    Args:
        graph_query_utils: GraphQueryUtils 实例
        
    Returns:
        Gradio应用实例
    """
    if gr is None:
        raise ModuleNotFoundError("gradio is required to launch the knowledge graph editor UI.")
    editor = KnowledgeGraphEditor(graph_query_utils)
    
    # 获取实体类型列表
    entity_types = editor.get_entity_types()
    
    with gr.Blocks(title="知识图谱编辑器") as app:
        gr.Markdown("# 🔍 知识图谱编辑器")

        gr.Markdown("支持实体查询、实体编辑和关系编辑功能")
        
        global_js = """
        <script>
        window.fillRelationFromEgo = function(centerId, otherId) {
            // 自动切换到「关系编辑」tab（如果不想自动切换，可以删掉这一小段）
            const tabs = document.querySelectorAll('button[id*="-tab-"]');
            const relTab = Array.from(tabs).find(t => t.textContent.includes('关系编辑'));
            if (relTab) relTab.click();

            setTimeout(() => {
                const labels = Array.from(document.querySelectorAll('label'));

                function findInputByLabelText(txt) {
                    const label = labels.find(l => l.textContent.includes(txt));
                    if (!label) return null;
                    // Textbox 可能是 textarea 也可能是 input，两种都试一下
                    let inp = label.parentElement.querySelector('textarea');
                    if (!inp) {
                        inp = label.parentElement.querySelector('input');
                    }
                    return inp;
                }

                const srcInput = findInputByLabelText('源实体ID');
                const tgtInput = findInputByLabelText('目标实体ID');

                // 按你的需求：源 = 中心节点ID；目标 = 邻居节点ID
                if (srcInput) {
                    srcInput.value = centerId || '';
                    srcInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
                if (tgtInput) {
                    tgtInput.value = otherId || '';
                    tgtInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }, 200);
        };
        </script>
        """
        gr.HTML(global_js)

        with gr.Tabs():
            # ==================== Tab 1: 实体查询 ====================
            with gr.Tab("🔎 实体查询"):
                gr.Markdown("### 搜索实体")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        search_keyword = gr.Textbox(
                            label="搜索关键词",
                            placeholder="输入实体名称或别名...",
                            lines=1
                        )
                    with gr.Column(scale=2):
                        search_type = gr.Dropdown(
                            label="实体类型（可选）",
                            choices=entity_types,
                            value="全部"
                        )
                    with gr.Column(scale=1):
                        search_btn = gr.Button("🔍 搜索", variant="primary")
                
                search_status = gr.Textbox(label="状态", interactive=False, lines=1)
                search_results = gr.HTML(label="搜索结果")
                
                search_btn.click(
                    fn=editor.search_entities,
                    inputs=[search_keyword, search_type],
                    outputs=[search_results, search_status]
                )
            
            # ==================== Tab 2: 实体编辑 ====================
            with gr.Tab("✏️ 实体编辑"):
                gr.Markdown("### 编辑实体信息")
                gr.Markdown("先输入实体ID加载，然后修改属性后点击保存")
                
                with gr.Row():
                    entity_id_input = gr.Textbox(
                        label="实体ID",
                        placeholder="例如: ent_37f43854cec8",
                        lines=1,
                        scale=3
                    )
                    load_entity_btn = gr.Button("📥 加载实体", variant="secondary", scale=1)
                
                entity_status = gr.Textbox(label="状态", interactive=False, lines=1)
                
                with gr.Row():
                    entity_name_input = gr.Textbox(
                        label="实体名称",
                        lines=1,
                        scale=2
                    )
                    entity_type_display = gr.Textbox(
                        label="实体类型（只读）",
                        lines=1,
                        interactive=False,
                        scale=1
                    )
                
                with gr.Row():
                    entity_description_input = gr.Textbox(
                        label="实体描述",
                        lines=3,
                        scale=3
                    )
                    embedding_status_display = gr.Textbox(
                        label="Embedding 状态",
                        lines=1,
                        interactive=False,
                        scale=1
                    )
                
                entity_aliases_input = gr.Code(
                    label="别名列表（JSON格式）",
                    language="json",
                    lines=5
                )
                
                entity_props_input = gr.Code(
                    label="实体属性（JSON格式，不包含 embedding 和 source_chunks）",
                    language="json",
                    lines=10
                )
                
                with gr.Row():
                    update_entity_btn = gr.Button("💾 保存修改", variant="primary", scale=1)
                    recalc_embedding_btn = gr.Button("🔄 重新计算 Embedding", variant="secondary", scale=1)
                    delete_entity_btn = gr.Button("🗑️ 删除实体", variant="stop", scale=1)
                
                # ==================== Ego Network 可视化 ====================
                gr.Markdown("---")
                gr.Markdown("### 🕸️ Ego Network 可视化")
                gr.Markdown("显示该节点及其 1-hop 邻居的网络图")
                
                with gr.Row():
                    visualize_btn = gr.Button("🔍 生成可视化", variant="primary", scale=1)
                
                ego_viz_status = gr.Textbox(label="可视化状态", interactive=False, lines=1)
                
                # 关系列表（可折叠）
                with gr.Accordion("📊 关系列表（点击行跳转到关系编辑）", open=False):
                    ego_relations_list = gr.HTML(label="关系列表")
                
                ego_viz_output = gr.HTML(label="Ego Network 图", elem_id="ego_network_viz")
                
                # 绑定事件
                load_entity_btn.click(
                    fn=editor.load_entity,
                    inputs=[entity_id_input],
                    outputs=[entity_name_input, entity_type_display, entity_description_input, 
                            entity_aliases_input, entity_props_input, embedding_status_display, entity_status]
                )
                
                update_entity_btn.click(
                    fn=editor.update_entity,
                    inputs=[entity_id_input, entity_name_input, entity_description_input,
                           entity_aliases_input, entity_props_input],
                    outputs=[entity_status]
                )
                
                recalc_embedding_btn.click(
                    fn=editor.recalculate_embedding,
                    inputs=[entity_id_input],
                    outputs=[embedding_status_display, entity_status]
                )
                
                delete_entity_btn.click(
                    fn=editor.delete_entity,
                    inputs=[entity_id_input],
                    outputs=[entity_status]
                )
                
                visualize_btn.click(
                    fn=editor.visualize_ego_network,
                    inputs=[entity_id_input],
                    outputs=[ego_relations_list, ego_viz_output, ego_viz_status]
                )
            
            # ==================== Tab 3: 关系编辑 ====================
            with gr.Tab("🔗 关系编辑"):
                gr.Markdown("### 编辑实体关系")
                gr.Markdown("输入源实体ID和目标实体ID，查看和编辑它们之间的关系")
                
                with gr.Row():
                    src_entity_id = gr.Textbox(
                        label="源实体ID",
                        placeholder="例如: ent_63a73deadb8e",
                        lines=1
                    )
                    tgt_entity_id = gr.Textbox(
                        label="目标实体ID",
                        placeholder="例如: ent_37f43854cec8",
                        lines=1
                    )
                    load_relations_btn = gr.Button("📥 加载关系", variant="secondary")
                
                relation_status = gr.Textbox(label="状态", interactive=False, lines=1)
                relation_results = gr.HTML(label="关系列表")
                
                gr.Markdown("### 修改关系属性")
                
                with gr.Row():
                    relation_type_input = gr.Textbox(
                        label="当前关系类型",
                        placeholder="例如: LIVES_IN",
                        lines=1,
                        scale=1
                    )
                    new_relation_type_input = gr.Textbox(
                        label="新关系类型（可选，不填则不修改）",
                        placeholder="例如: LOCATED_AT",
                        lines=1,
                        scale=1
                    )
                
                relation_props_input = gr.Code(
                    label="关系属性（JSON格式）",
                    language="json",
                    lines=10,
                    value="{}"
                )
                
                with gr.Row():
                    update_relation_btn = gr.Button("💾 保存关系修改", variant="primary", scale=1)
                    delete_relation_btn = gr.Button("🗑️ 删除关系", variant="stop", scale=1)
                
                load_relations_btn.click(
                    fn=editor.load_relations,
                    inputs=[src_entity_id, tgt_entity_id],
                    outputs=[relation_results, relation_status, relation_type_input, relation_props_input]
                )
                
                update_relation_btn.click(
                    fn=editor.update_relation,
                    inputs=[src_entity_id, tgt_entity_id, relation_type_input, new_relation_type_input, relation_props_input],
                    outputs=[relation_status]
                )
                
                delete_relation_btn.click(
                    fn=editor.delete_relation,
                    inputs=[src_entity_id, tgt_entity_id, relation_type_input],
                    outputs=[relation_status]
                )
        
        gr.Markdown("---")
        gr.Markdown("💡 **使用提示**: 在实体查询中找到实体ID后，可以直接点击表格行跳转到编辑页面")
    
    return app


# ==================== 主程序入口 ====================

def launch_editor(graph_query_utils, share=False, server_port=7860):
    """
    启动知识图谱编辑器
    
    Args:
        graph_query_utils: GraphQueryUtils 实例
        share: 是否创建公共链接
        server_port: 服务器端口
    """
    app = create_gradio_interface(graph_query_utils)
    
    # 确保静态文件目录存在
    static_dir = "./kg_editor/static"
    os.makedirs(static_dir, exist_ok=True)
    
    # 启动应用，允许访问静态文件目录
    app.launch(
        share=share, 
        server_port=server_port, 
        server_name="0.0.0.0",
        allowed_paths=[static_dir]
    )


if __name__ == "__main__":
    print("请在您的代码中导入并使用 launch_editor 函数")
    print("示例代码已在文件注释中提供")
