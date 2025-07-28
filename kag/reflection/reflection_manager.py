# kag/reflection/reflection_manager.py

import os
import json
import time
from typing import Dict, Any, List, Optional

from kag.utils.config import ReflectionConfig

class ReflectionManager:
    """反思管理器
    
    负责管理Agent的反思过程，包括：
    - 记录任务执行情况
    - 分析成功和失败的原因
    - 生成改进建议
    - 保存和加载反思记录
    """
    
    def __init__(self, config: ReflectionConfig):
        """初始化反思管理器
        
        Args:
            config: 反思配置
        """
        self.config = config
        self.reflections: List[Dict[str, Any]] = []
        self.task_count = 0
        self.reflection_path = os.path.join(config.reflection_path, "reflections.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.reflection_path), exist_ok=True)
        
        # 加载已有的反思记录
        self.load()
    
    def record_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """记录任务执行情况
        
        Args:
            task: 任务信息
            result: 任务结果
        """
        if not self.config.enabled:
            return
            
        self.task_count += 1
        
        # 检查是否需要进行反思
        if self.task_count % self.config.reflection_interval == 0:
            self.reflect(task, result)
    
    def reflect(self, task: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """进行反思
        
        Args:
            task: 最近的任务信息
            result: 最近的任务结果
            
        Returns:
            反思结果
        """
        if not self.config.enabled:
            return {}
            
        try:
            # 创建反思记录
            reflection = {
                "timestamp": time.time(),
                "task_count": self.task_count,
                "task": task,
                "result": result,
                "analysis": self._analyze_performance(task, result),
                "suggestions": self._generate_suggestions(task, result)
            }
            
            # 添加到反思列表
            self.reflections.append(reflection)
            
            # 保持最大反思数量
            if len(self.reflections) > self.config.max_reflections:
                self.reflections = self.reflections[-self.config.max_reflections:]
            
            # 保存反思记录
            self.save()
            
            print(f"完成第 {len(self.reflections)} 次反思")
            return reflection
        except Exception as e:
            print(f"反思过程失败: {str(e)}")
            return {}
    
    def reflect_with_llm(self, llm: Any, task: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用LLM进行反思
        
        Args:
            llm: 语言模型
            task: 最近的任务信息
            result: 最近的任务结果
            
        Returns:
            反思结果
        """
        if not self.config.enabled:
            return {}
            
        try:
            # 构建反思提示词
            prompt = self._build_reflection_prompt(task, result)
            
            # 调用LLM进行反思
            if hasattr(llm, "invoke"):
                # LangChain风格
                response = llm.invoke(prompt)
                reflection_text = response.content if hasattr(response, "content") else str(response)
            elif hasattr(llm, "chat"):
                # Qwen-Agent风格
                messages = [
                    {"role": "system", "content": "你是一个智能反思助手，擅长分析任务执行情况并提出改进建议。"},
                    {"role": "user", "content": prompt}
                ]
                reflection_text = llm.chat(messages)
            else:
                # 备用方案
                reflection_text = "无法使用LLM进行反思"
            
            # 创建反思记录
            reflection = {
                "timestamp": time.time(),
                "task_count": self.task_count,
                "task": task,
                "result": result,
                "llm_reflection": reflection_text,
                "analysis": self._analyze_performance(task, result),
                "suggestions": self._generate_suggestions(task, result)
            }
            
            # 添加到反思列表
            self.reflections.append(reflection)
            
            # 保持最大反思数量
            if len(self.reflections) > self.config.max_reflections:
                self.reflections = self.reflections[-self.config.max_reflections:]
            
            # 保存反思记录
            self.save()
            
            print(f"使用LLM完成第 {len(self.reflections)} 次反思")
            return reflection
        except Exception as e:
            print(f"LLM反思过程失败: {str(e)}")
            # 回退到简单反思
            return self.reflect(task, result)
    
    def get_reflections(self, k: int = 5) -> List[Dict[str, Any]]:
        """获取最近的反思记录
        
        Args:
            k: 返回的反思记录数量
            
        Returns:
            反思记录列表
        """
        return self.reflections[-k:]
    
    def get_suggestions(self) -> List[str]:
        """获取所有改进建议
        
        Returns:
            改进建议列表
        """
        suggestions = []
        for reflection in self.reflections:
            if "suggestions" in reflection:
                suggestions.extend(reflection["suggestions"])
        return suggestions
    
    def clear(self) -> None:
        """清空反思记录"""
        self.reflections = []
        self.task_count = 0
        self.save()
    
    def save(self) -> None:
        """保存反思记录到磁盘"""
        try:
            data = {
                "reflections": self.reflections,
                "task_count": self.task_count
            }
            with open(self.reflection_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存反思记录失败: {str(e)}")
    
    def load(self) -> None:
        """从磁盘加载反思记录"""
        if os.path.exists(self.reflection_path):
            try:
                with open(self.reflection_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.reflections = data.get("reflections", [])
                    self.task_count = data.get("task_count", 0)
            except Exception as e:
                print(f"加载反思记录失败: {str(e)}")
                self.reflections = []
                self.task_count = 0
        else:
            self.reflections = []
            self.task_count = 0
    
    def _analyze_performance(self, task: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析任务执行性能
        
        Args:
            task: 任务信息
            result: 任务结果
            
        Returns:
            性能分析结果
        """
        analysis = {
            "success": False,
            "error_count": 0,
            "execution_time": 0,
            "quality_score": 0
        }
        
        if result:
            # 判断任务是否成功
            analysis["success"] = result.get("status") == "completed"
            
            # 统计错误
            if "error" in result:
                analysis["error_count"] = 1
            
            # 评估质量（简化处理）
            if analysis["success"]:
                analysis["quality_score"] = 0.8
            else:
                analysis["quality_score"] = 0.2
        
        return analysis
    
    def _generate_suggestions(self, task: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> List[str]:
        """生成改进建议
        
        Args:
            task: 任务信息
            result: 任务结果
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        if result:
            if result.get("status") == "failed":
                suggestions.append("任务执行失败，需要检查错误原因")
                
                if "error" in result:
                    error_msg = result["error"]
                    if "timeout" in error_msg.lower():
                        suggestions.append("考虑增加超时时间")
                    elif "memory" in error_msg.lower():
                        suggestions.append("考虑优化内存使用")
                    elif "network" in error_msg.lower():
                        suggestions.append("检查网络连接")
                    else:
                        suggestions.append("分析具体错误信息并修复")
            else:
                suggestions.append("任务执行成功，继续保持")
        
        # 基于历史反思生成建议
        if len(self.reflections) > 1:
            recent_failures = sum(1 for r in self.reflections[-5:] 
                                if r.get("analysis", {}).get("success") is False)
            if recent_failures > 2:
                suggestions.append("最近失败率较高，需要全面检查系统")
        
        return suggestions
    
    def _build_reflection_prompt(self, task: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> str:
        """构建反思提示词
        
        Args:
            task: 任务信息
            result: 任务结果
            
        Returns:
            反思提示词
        """
        prompt = f"""请对以下任务执行情况进行反思和分析：

任务信息：
{json.dumps(task, ensure_ascii=False, indent=2) if task else "无"}

执行结果：
{json.dumps(result, ensure_ascii=False, indent=2) if result else "无"}

历史反思记录（最近5次）：
{json.dumps(self.get_reflections(5), ensure_ascii=False, indent=2)}

请分析：
1. 任务执行的成功和失败原因
2. 可能的改进点
3. 对未来任务的建议

请提供具体、可操作的建议。
"""
        return prompt

