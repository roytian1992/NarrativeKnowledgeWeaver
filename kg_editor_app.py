"""
知识图谱编辑器 - Gradio 应用
提供实体查询、实体编辑和关系编辑功能
"""

import gradio as gr
import json
from typing import List, Dict, Any, Optional, Tuple
from pyvis.network import Network
import tempfile
import os
import base64


class KnowledgeGraphEditor:
    """知识图谱编辑器核心类"""
    
    def __init__(self, neo4j_utils):
        """
        初始化编辑器
        
        Args:
            neo4j_utils: Neo4jUtils 实例
        """
        self.neo4j_utils = neo4j_utils
    
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
            # 构造查询条件
            if not keyword and not entity_type:
                return "<p style='color: orange;'>请输入搜索关键词或选择实体类型</p>", "⚠️ 请提供搜索条件"
            
            # 构造 MATCH 子句
            if entity_type and entity_type != "全部":
                match_clause = f"MATCH (e:{entity_type})"
            else:
                match_clause = "MATCH (e)"
            
            # 构造 WHERE 子句
            where_clauses = []
            params = {}
            
            if keyword:
                where_clauses.append(
                    "(e.name CONTAINS $kw OR any(alias IN e.aliases WHERE alias CONTAINS $kw))"
                )
                params["kw"] = keyword
            
            where_clause = ""
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses)
            
            # 完整查询
            cypher = f"""
            {match_clause}
            {where_clause}
            RETURN e.id as id, e.name as name, labels(e) as types, e.description as description
            LIMIT 50
            """
            
            results = self.neo4j_utils.execute_query(cypher, params)
            
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
            cypher = """
            MATCH (e)
            RETURN DISTINCT labels(e) as types
            """
            results = self.neo4j_utils.execute_query(cypher)
            
            types = set()
            for record in results:
                for label in record.get('types', []):
                    types.add(label)
            
            return ["全部"] + sorted(list(types))
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
            
            cypher = """
            MATCH (e {id: $entity_id})
            RETURN e
            """
            
            results = self.neo4j_utils.execute_query(cypher, {"entity_id": entity_id.strip()})
            
            if not results:
                return "", "", "", "", "", "", f"❌ 未找到实体: {entity_id}"
            
            entity_node = results[0]['e']
            
            # 提取实体信息
            entity_name = entity_node.get('name', '')
            entity_description = entity_node.get('description', '')
            
            # 处理 labels 属性
            if hasattr(entity_node, 'labels'):
                entity_types = ', '.join(entity_node.labels)
            else:
                entity_types = 'Unknown'
            
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

            if hasattr(rel, "type"):
                rel_type = rel.type
            else:
                rel_type = "RELATED_TO"

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

            if hasattr(rel, "type"):
                rel_type = rel.type
            else:
                rel_type = "RELATED_TO"

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
            if not entity_id or not entity_id.strip():
                # 注意：这里要返回 3 个值，对应 gradio 的 3 个输出
                return "", "", "⚠️ 请输入实体ID"

            # 查询中心节点和 1-hop 邻居
            cypher = """
            MATCH (center {id: $entity_id})
            OPTIONAL MATCH (center)-[r1]->(neighbor)
            OPTIONAL MATCH (other)-[r2]->(center)
            RETURN center, 
                   collect(DISTINCT {node: neighbor, rel: r1, direction: 'out'}) as outgoing,
                   collect(DISTINCT {node: other, rel: r2, direction: 'in'}) as incoming
            """

            results = self.neo4j_utils.execute_query(cypher, {"entity_id": entity_id.strip()})

            if not results:
                return "", "", f"❌ 未找到实体: {entity_id}"

            result = results[0]
            center_node = result["center"]
            outgoing = result.get("outgoing", [])
            incoming = result.get("incoming", [])

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
            if hasattr(center_node, "labels"):
                center_type = ", ".join(center_node.labels)
            else:
                center_type = "Unknown"

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

                if hasattr(neighbor, "labels"):
                    neighbor_type = ", ".join(neighbor.labels)
                else:
                    neighbor_type = "Unknown"

                net.add_node(
                    neighbor_id,
                    label=neighbor_name,
                    title=f"ID: {neighbor_id}\nType: {neighbor_type}",
                    color="#4ecdc4",
                    size=20,
                )

                if rel is not None:
                    if hasattr(rel, "type"):
                        rel_type = rel.type
                    else:
                        rel_type = "RELATED_TO"

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

                if hasattr(other, "labels"):
                    other_type = ", ".join(other.labels)
                else:
                    other_type = "Unknown"

                if other_id not in [node["id"] for node in net.nodes]:
                    net.add_node(
                        other_id,
                        label=other_name,
                        title=f"ID: {other_id}\nType: {other_type}",
                        color="#95e1d3",
                        size=20,
                    )

                if rel is not None:
                    if hasattr(rel, "type"):
                        rel_type = rel.type
                    else:
                        rel_type = "RELATED_TO"

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
            
            # 解析属性JSON
            try:
                properties = json.loads(properties_json) if properties_json else {}
            except json.JSONDecodeError as e:
                return f"❌ 属性JSON格式错误: {str(e)}"
            
            # 构造更新参数
            params = {"entity_id": entity_id.strip()}
            set_clauses = []
            
            # 更新基本字段
            if entity_name:
                set_clauses.append("e.name = $name")
                params["name"] = entity_name
            
            if entity_description:
                set_clauses.append("e.description = $description")
                params["description"] = entity_description
            
            if aliases:
                set_clauses.append("e.aliases = $aliases")
                params["aliases"] = aliases
            
            # 更新 properties
            if properties:
                set_clauses.append("e.properties = $properties")
                params["properties"] = json.dumps(properties, ensure_ascii=False)
                
                # 同时将 properties 中的字段直接设置到节点上
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    set_clauses.append(f"e.{key} = ${param_name}")
                    params[param_name] = value
            
            if not set_clauses:
                return "⚠️ 没有要更新的内容"
            
            cypher = f"""
            MATCH (e {{id: $entity_id}})
            SET {', '.join(set_clauses)}
            RETURN e
            """
            
            results = self.neo4j_utils.execute_query(cypher, params)
            
            if not results:
                return f"❌ 未找到实体: {entity_id}"
            
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
            
            # 先检查实体是否存在
            check_cypher = """
            MATCH (e {id: $entity_id})
            RETURN e.name as name
            """
            
            results = self.neo4j_utils.execute_query(check_cypher, {"entity_id": entity_id.strip()})
            
            if not results:
                return f"❌ 未找到实体: {entity_id}"
            
            entity_name = results[0].get('name', entity_id)
            
            # 删除实体及其所有关系
            delete_cypher = """
            MATCH (e {id: $entity_id})
            DETACH DELETE e
            """
            
            self.neo4j_utils.execute_query(delete_cypher, {"entity_id": entity_id.strip()})
            
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
            
            # 加载实体
            cypher = """
            MATCH (e {id: $entity_id})
            RETURN e
            """
            
            results = self.neo4j_utils.execute_query(cypher, {"entity_id": entity_id.strip()})
            
            if not results:
                return "", f"❌ 未找到实体: {entity_id}"
            
            entity_node = results[0]['e']
            
            # 提取实体类型
            if hasattr(entity_node, 'labels'):
                entity_types = list(entity_node.labels)
                node_type = entity_types[0] if entity_types else ""
            else:
                node_type = ""
            
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
            
            # 调用 neo4j_utils 的 encode_node_embedding 方法
            if not hasattr(self.neo4j_utils, 'encode_node_embedding'):
                return "", "❌ Neo4jUtils 不支持 encode_node_embedding 方法"
            
            embedding = self.neo4j_utils.encode_node_embedding(node_data)
            
            # 更新 embedding
            update_cypher = """
            MATCH (e {id: $entity_id})
            SET e.embedding = $embedding
            RETURN e
            """
            
            self.neo4j_utils.execute_query(update_cypher, {
                "entity_id": entity_id.strip(),
                "embedding": embedding
            })
            
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
            
            # 双向查询：查找 src->tgt 和 tgt->src 的关系
            cypher = """
            MATCH (src {id: $src_id})-[r]->(tgt {id: $tgt_id})
            RETURN src.name as src_name, src.id as src_id, type(r) as rel_type, 
                   properties(r) as rel_props, tgt.name as tgt_name, tgt.id as tgt_id, 'forward' as direction
            UNION
            MATCH (tgt {id: $src_id})<-[r]-(src {id: $tgt_id})
            RETURN src.name as src_name, src.id as src_id, type(r) as rel_type, 
                   properties(r) as rel_props, tgt.name as tgt_name, tgt.id as tgt_id, 'reverse' as direction
            """
            
            results = self.neo4j_utils.execute_query(cypher, {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            })
            
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
            
            # 判断是否需要修改关系类型
            if new_rel_type and new_rel_type.strip() and new_rel_type.strip() != rel_type.strip():
                # 需要修改关系类型：删除旧关系，创建新关系
                new_type = new_rel_type.strip()
                
                params = {
                    "src_id": src_entity_id.strip(),
                    "tgt_id": tgt_entity_id.strip(),
                    "properties": json.dumps(properties, ensure_ascii=False)
                }
                
                # 添加属性参数
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    params[param_name] = value
                
                # 构造属性设置子句
                prop_assignments = ["properties: $properties"]
                for key in properties.keys():
                    prop_assignments.append(f"{key}: $prop_{key}")
                
                cypher = f"""
                MATCH (src {{id: $src_id}})-[old_r:{rel_type}]->(tgt {{id: $tgt_id}})
                CREATE (src)-[new_r:{new_type}]->(tgt)
                SET new_r = {{{', '.join(prop_assignments)}}}
                DELETE old_r
                RETURN new_r
                """
                
                results = self.neo4j_utils.execute_query(cypher, params)
                
                if not results:
                    return f"❌ 未找到指定的关系: {rel_type}"
                
                return f"✅ 关系类型已修改: {src_entity_id} -[{rel_type}]-> {tgt_entity_id} → {src_entity_id} -[{new_type}]-> {tgt_entity_id}"
            
            else:
                # 只更新属性，不修改关系类型
                params = {
                    "src_id": src_entity_id.strip(),
                    "tgt_id": tgt_entity_id.strip(),
                    "properties": json.dumps(properties, ensure_ascii=False)
                }
                
                # 将属性保存到 properties 字段
                set_clauses = ["r.properties = $properties"]
                
                # 同时也将属性展开到关系的顶层（保持兼容性）
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    set_clauses.append(f"r.{key} = ${param_name}")
                    params[param_name] = value
                
                cypher = f"""
                MATCH (src {{id: $src_id}})-[r:{rel_type}]->(tgt {{id: $tgt_id}})
                SET {', '.join(set_clauses)}
                RETURN r
                """
                
                results = self.neo4j_utils.execute_query(cypher, params)
                
                if not results:
                    return f"❌ 未找到指定的关系: {rel_type}"
                
                return f"✅ 关系更新成功: {src_entity_id} -[{rel_type}]-> {tgt_entity_id}"
            
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
            
            # 先检查关系是否存在
            check_cypher = f"""
            MATCH (src {{id: $src_id}})-[r:{rel_type}]->(tgt {{id: $tgt_id}})
            RETURN src.name as src_name, tgt.name as tgt_name
            """
            
            results = self.neo4j_utils.execute_query(check_cypher, {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            })
            
            if not results:
                return f"❌ 未找到指定的关系: {rel_type}"
            
            src_name = results[0].get('src_name', src_entity_id)
            tgt_name = results[0].get('tgt_name', tgt_entity_id)
            
            # 删除关系
            delete_cypher = f"""
            MATCH (src {{id: $src_id}})-[r:{rel_type}]->(tgt {{id: $tgt_id}})
            DELETE r
            """
            
            self.neo4j_utils.execute_query(delete_cypher, {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            })
            
            return f"✅ 关系已删除: {src_name} -[{rel_type}]-> {tgt_name}"
            
        except Exception as e:
            return f"❌ 删除出错: {str(e)}"
    
    def get_relation_types(self, src_entity_id: str, tgt_entity_id: str) -> List[str]:
        """获取两个实体之间的关系类型"""
        try:
            if not src_entity_id or not tgt_entity_id:
                return [""]
            
            cypher = """
            MATCH (src {id: $src_id})-[r]->(tgt {id: $tgt_id})
            RETURN DISTINCT type(r) as rel_type
            """
            
            results = self.neo4j_utils.execute_query(cypher, {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            })
            
            return [record['rel_type'] for record in results]
        except Exception as e:
            print(f"获取关系类型出错: {e}")
            return [""]


def create_gradio_interface(neo4j_utils):
    """
    创建Gradio界面
    
    Args:
        neo4j_utils: Neo4jUtils 实例
        
    Returns:
        Gradio应用实例
    """
    editor = KnowledgeGraphEditor(neo4j_utils)
    
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

def launch_editor(neo4j_utils, share=False, server_port=7860):
    """
    启动知识图谱编辑器
    
    Args:
        neo4j_utils: Neo4jUtils 实例
        share: 是否创建公共链接
        server_port: 服务器端口
    """
    app = create_gradio_interface(neo4j_utils)
    
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
