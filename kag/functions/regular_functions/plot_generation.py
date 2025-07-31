"""
Plot生成功能模块
用于从事件聚类生成情节单元
"""

from typing import Dict, Any, List
import json
import logging
from kag.utils.format import is_valid_json, correct_json_format


class PlotGenerator:
    """Plot生成器类"""
    
    def __init__(self, prompt_loader=None, llm=None):
        """
        初始化Plot生成器
        
        Args:
            prompt_loader: 提示词加载器
            llm: 大语言模型实例
        """
        super().__init__()
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.logger = logging.getLogger(__name__)
    
    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成Plot单元
        
        Args:
            params: 参数字典，包含：
                - event_cluster: 事件聚类列表
                - event_details: 事件详细信息
                - causality_paths: 因果关系路径
                
        Returns:
            Dict: 生成的Plot单元数据
        """
        try:
            event_cluster = params.get("event_cluster", [])
            event_details = params.get("event_details", [])
            causality_paths = params.get("causality_paths", [])
            
            if not event_cluster:
                return {"error": "事件聚类为空"}
            
            # 准备事件详细信息文本
            event_details_text = self._format_event_details(event_details)
            
            # 准备因果关系文本
            causality_text = self._format_causality_paths(causality_paths)
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id='plot_construction_prompt',
                variables={
                    'event_details': event_details_text,
                    'causality_paths': causality_text
                }
            )
            
            # 调用LLM生成Plot
            messages = [
                {"role": "system", "content": "你是一个专业的叙事分析专家。"},
                {"role": "user", "content": prompt_text}
            ]
            
            response = self.llm.invoke(messages)
            
            # 解析响应
            plot_data = self._parse_plot_response(response.content)
            
            if plot_data:
                # 确保event_ids包含聚类中的所有事件
                plot_data["event_ids"] = event_cluster
                self.logger.info(f"成功生成Plot单元: {plot_data.get('title', 'Unknown')}")
                return plot_data
            else:
                self.logger.error("Plot生成失败：无法解析LLM响应")
                return {"error": "Plot生成失败"}
                
        except Exception as e:
            self.logger.error(f"Plot生成过程中发生错误: {e}")
            return {"error": f"Plot生成失败: {str(e)}"}
    
    def _format_event_details(self, event_details: List[Dict[str, Any]]) -> str:
        """
        格式化事件详细信息为文本
        
        Args:
            event_details: 事件详细信息列表
            
        Returns:
            str: 格式化后的事件信息文本
        """
        if not event_details:
            return "无事件详细信息"
        
        formatted_events = []
        for event in event_details:
            event_text = f"事件ID: {event.get('event_id', 'Unknown')}\n"
            event_text += f"事件名称: {event.get('event_name', 'Unknown')}\n"
            event_text += f"事件描述: {event.get('event_description', 'Unknown')}\n"
            
            participants = event.get('participants', [])
            if participants:
                event_text += f"参与者: {', '.join(participants)}\n"
            
            scene_names = event.get('scene_names', [])
            if scene_names:
                event_text += f"所属场景: {', '.join(scene_names)}\n"
            
            location = event.get('location')
            if location:
                event_text += f"地点: {location}\n"
                
            time = event.get('time')
            if time:
                event_text += f"时间: {time}\n"
            
            formatted_events.append(event_text)
        
        return "\n---\n".join(formatted_events)
    
    def _format_causality_paths(self, causality_paths: List[Dict[str, Any]]) -> str:
        """
        格式化因果关系路径为文本
        
        Args:
            causality_paths: 因果关系路径列表
            
        Returns:
            str: 格式化后的因果关系文本
        """
        if not causality_paths:
            return "无因果关系信息"
        
        formatted_paths = []
        for path in causality_paths:
            source_name = path.get('source_name', path.get('source_id', 'Unknown'))
            target_name = path.get('target_name', path.get('target_id', 'Unknown'))
            weight = path.get('weight', 'Unknown')
            description = path.get('causality_description', '')
            
            path_text = f"{source_name} → {target_name} (权重: {weight})"
            if description:
                path_text += f"\n  因果描述: {description}"
            
            formatted_paths.append(path_text)
        
        return "\n".join(formatted_paths)
    
    def _parse_plot_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取Plot数据
        
        Args:
            response: LLM响应文本
            
        Returns:
            Dict: 解析后的Plot数据，失败时返回None
        """
        try:
            # 尝试直接解析JSON
            if is_valid_json(response):
                return json.loads(response)
            
            # 尝试修正JSON格式
            corrected_json = correct_json_format(response)
            if corrected_json and is_valid_json(corrected_json):
                return json.loads(corrected_json)
            
            # 尝试从响应中提取JSON部分
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                if is_valid_json(json_str):
                    return json.loads(json_str)
            
            # 尝试查找花括号包围的JSON
            brace_pattern = r'\{.*\}'
            brace_match = re.search(brace_pattern, response, re.DOTALL)
            
            if brace_match:
                json_str = brace_match.group(0)
                if is_valid_json(json_str):
                    return json.loads(json_str)
            
            self.logger.error(f"无法解析Plot响应: {response[:200]}...")
            return None
            
        except Exception as e:
            self.logger.error(f"解析Plot响应时发生错误: {e}")
            return None
    

