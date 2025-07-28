# kag/agent/qwen3_agent.py
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator, Literal
from qwen_agent.llm.schema import Message, FUNCTION
import copy
from kag.llm.llm_manager import LLMManager
from kag.llm.qwen3_llm import NonStreamingFnCallAgent
from kag.utils.prompt_loader import PromptLoader
from kag.builder.tools_loader import load_all_tools
from kag.agent.base_agent import BaseAgent
from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS 
# 导入Qwen-Agent的类
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM


def task_parser(task: Dict[str, Any]) -> str:
    """将任务字典转换为用户输入字符串"""
    task_map = {
        "text": "文本",
        "goal": "目标",
        "entity_type_description_text": "实体类型描述",
        "relation_type_description_text": "关系类型描述",
        "scene_name": "场景名称",
        "entities": "实体列表",
        "relations": "关系列表",
    }

    user_input = []
    for key in task:
        if key in task_map:
            value = task[key]
            if isinstance(value, list):
                value = "\n".join(value)
            elif isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            user_input.append(f"{task_map[key]}：{value}")
        else:
            user_input.append(f"{key}：{task[key]}")
            
    return "\n\n".join(user_input)


class Qwen3Agent(BaseAgent):
    """Qwen3 Agent实现"""
    
    def __init__(self, config, llm):
        """初始化Agent
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        
        # # 初始化LLM
        # self.llm_manager = LLMManager(config)
        # self.llm = self.llm_manager.get_llm()
        self.llm = llm
        
        # 加载提示词 - 修复：传入prompt_dir字符串而不是整个config对象
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
        self.prompt_loader = PromptLoader(prompt_dir)
        self.agent_prompt = self.prompt_loader.load_prompt("agent_prompt")
        
        # 加载工具 - 确保工具是字典格式
        # self.tools = []

        self.tools = load_all_tools(
            self.prompt_loader,
            self.llm
        )
        self.tools_map = {tool.name: tool for tool in self.tools}

        print("[CHECK] Tools: ", self.tools)
            
        # 创建FnCallAgent
        self.agent = NonStreamingFnCallAgent(function_list=self.tools, llm=self.llm)

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务
        
        Args:
            task: 任务字典，包含目标和文本
            
        Returns:
            任务结果字典
        """
        try:
            user_input = task_parser(task)
            extraction_goal = task.get('goal', '')
            goal_list = ["抽取实体", "抽取关系", "抽取剧本元素", "抽取道具", "解析场景名称", "评估抽取质量", "反思"]
            for goal in goal_list:
                if goal in extraction_goal:
                    return self.direct_tool_call(task)
            
            user_message = Message(role=USER, content=user_input)

            # ② 一定要用 list 包起来
            messages = [user_message]

            all_responses = self.agent._run(messages=messages, stream=False, lang="zh")
            # print("[CHECK] all_responses:", all_responses)

            # 提取最后一次的文本
            result_content = ""
            if all_responses:
                for msg in all_responses:
                    if isinstance(msg, dict) and msg.get("role") == "function":
                        result_content += msg.get("content", "")
                    elif hasattr(msg, "role"):
                        result_content += msg.content

            return {"status": "completed", "result": result_content}

        except Exception as e:
            print(f"Error in Qwen3Agent.run: {str(e)}")
            return {"status": "failed", "error": str(e)}


    def direct_tool_call(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            extraction_goal = task.pop("goal")
            match = re.search(r'[a-zA-Z_]+', extraction_goal)
            tool_name = match.group()
            tool = self.tools_map.get(tool_name)
            params = json.dumps(task)
            # print(f"[CHECK] 手动调用工具 {tool.name}")
            result = tool.call(params)
            return {"status": "completed", "result": result}

        except Exception as e:
            print(f"Error in Qwen3Agent.run: {str(e)}")
            return {"status": "failed", "error": str(e)}
            
    
