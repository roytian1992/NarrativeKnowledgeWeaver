"""
知识图谱编辑器 - Gradio 应用
提供实体查询、实体编辑和关系编辑功能
"""

import gradio as gr
import json
from typing import List, Dict, Any, Optional, Tuple


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
                return "<p style='color: orange;'>请输入源实体ID和目标实体ID</p>", "⚠️ 请输入完整的实体ID"
            
            cypher = """
            MATCH (src {id: $src_id})-[r]->(tgt {id: $tgt_id})
            RETURN src.name as src_name, type(r) as rel_type, properties(r) as rel_props, tgt.name as tgt_name
            """
            
            results = self.neo4j_utils.execute_query(cypher, {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            })
            
            if not results:
                return "<p style='color: gray;'>未找到这两个实体之间的关系</p>", "ℹ️ 未找到关系"
            
            # 构造结果表格
            html = f"""
            <div style='margin-bottom: 10px;'>
                <strong>源实体:</strong> {results[0]['src_name']} ({src_entity_id})<br>
                <strong>目标实体:</strong> {results[0]['tgt_name']} ({tgt_entity_id})
            </div>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background-color: #f0f0f0;'>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系类型</th>
                        <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>关系属性</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for record in results:
                rel_type = record.get('rel_type', 'N/A')
                rel_props = record.get('rel_props', {})
                rel_props_str = json.dumps(rel_props, ensure_ascii=False, indent=2) if rel_props else "{}"
                
                html += f"""
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 8px;'><strong>{rel_type}</strong></td>
                        <td style='border: 1px solid #ddd; padding: 8px;'><pre style='margin: 0;'>{rel_props_str}</pre></td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
            
            status = f"✅ 找到 {len(results)} 个关系"
            return html, status
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>查询出错: {str(e)}</p>"
            return error_msg, f"❌ 错误: {str(e)}"
    
    def update_relation(self, src_entity_id: str, tgt_entity_id: str, 
                       rel_type: str, properties_json: str) -> str:
        """
        更新关系属性
        
        Args:
            src_entity_id: 源实体ID
            tgt_entity_id: 目标实体ID
            rel_type: 关系类型
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
            
            # 构造SET子句
            set_clauses = []
            params = {
                "src_id": src_entity_id.strip(),
                "tgt_id": tgt_entity_id.strip()
            }
            
            for key, value in properties.items():
                param_name = f"prop_{key}"
                set_clauses.append(f"r.{key} = ${param_name}")
                params[param_name] = value
            
            if not set_clauses:
                return "⚠️ 没有要更新的属性"
            
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
                        label="关系类型",
                        placeholder="例如: LIVES_IN",
                        lines=1,
                        scale=2
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
                    outputs=[relation_results, relation_status]
                )
                
                update_relation_btn.click(
                    fn=editor.update_relation,
                    inputs=[src_entity_id, tgt_entity_id, relation_type_input, relation_props_input],
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
    app.launch(share=share, server_port=server_port, server_name="0.0.0.0")


if __name__ == "__main__":
    print("请在您的代码中导入并使用 launch_editor 函数")
    print("示例代码已在文件注释中提供")
