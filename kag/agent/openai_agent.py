# kag/agent/openai_agent.py

import json
from typing import Dict, Any

from kag.llm.llm_manager import LLMManager
from kag.utils.prompt_loader import PromptLoader
from kag.schema.kg_schema import ENTITY_TYPES
from kag.builder.tools_loader import load_all_tools
from kag.agent.base_agent import BaseAgent

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class OpenAIAgent(BaseAgent):
    """基于 OpenAI Functions Agent 的知识图谱信息抽取 Agent"""

    def __init__(self, config, llm):
        """初始化Agent
        
        Args:
            config: 配置
        """
        super().__init__(config)

        # 初始化 LLM (GPT 版本，直接用 ChatOpenAI)
        self.llm = llm
        # self.llm = ChatOpenAI(
        #     model=config.llm.model_name,
        #     temperature=config.llm.temperature,
        #     openai_api_key=config.llm.api_key
        # )

        # 初始化 PromptLoader
        self.prompt_loader = PromptLoader(
            prompt_dir='kag/prompts',
            global_variables={}
        )

        # 动态准备 entity_types_list / relation_types_list
        self.entity_types_list = [e['type'] for e in ENTITY_TYPES]
        
        from kag.schema.kg_schema import RELATION_TYPE_GROUPS
        self.relation_types_list = [
            r['type']
            for group in RELATION_TYPE_GROUPS.values()
            for r in group
        ]

        # 初始化 Tools
        self.tools = load_all_tools(
            self.prompt_loader,
            self.entity_types_list,
            self.relation_types_list,
            self.llm
        )

        # 初始化 OpenAI Functions Agent
        agent_prompt_data = self.prompt_loader.load_prompt("agent_prompt")
        agent_prompt = PromptTemplate.from_template(agent_prompt_data["template"])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=agent_prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行 OpenAI Functions Agent 执行信息抽取任务

        Args:
            task: dict, 例如 {"text": "xxx", "extraction_goal": "xxx"}

        Returns:
            运行结果
        """
        try:
            agent_input = f"请根据以下文本执行知识图谱信息抽取任务：{task.get('text', '')}。\n任务目标：{task.get('extraction_goal', '')}"
            result = self.agent_executor.run(
                task=json.dumps(task),
                input=agent_input
            )
            return {
                "status": "completed",
                "result": result
            }
        except Exception as e:
            error_msg = f"OpenAI Agent 执行失败: {e}"
            print(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }

