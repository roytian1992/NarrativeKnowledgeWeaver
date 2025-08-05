# kag/tools/qwen_tools/entity_extraction_tool.py

from typing import Dict, Any, List
import json
from kag.utils.format import correct_json_format, is_valid_json
from qwen_agent.tools.base import BaseTool, register_tool
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseChain


class WrappedQwenLLM(LLM):
    def __init__(self, qwen_fncall_llm):
        super().__init__()
        object.__setattr__(self, "_qwen", qwen_fncall_llm)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [Message(role=USER, content=prompt)]
        responses = self._qwen._chat_no_stream(messages, generate_cfg={
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "enable_thinking": False  # ✅ 禁用思考过程
        })
        return responses[0].content

    @property
    def _llm_type(self) -> str:
        return "wrapped-qwen-fncall"



@register_tool("sql_database")
class QwenSQLDatabaseTool(BaseTool):
    """查询SQL数据库，回答与对话相关的问题"""
    
    name = "sql_database"
    description = "查询SQL数据库，回答与剧本中人物对话相关的内容"
    parameters = [
        {
            "name": "question",
            "type": "string",
            "description": "需要通过查询数据库回答的问题",
            "required": True
        }
    ]
    
    def __init__(self, db_path, llm=None):
        """初始化工具
        
        Args:
            db_path: 数据库路径
            llm: 语言模型
        """
        super().__init__()
        self.db = SQLDatabase.from_uri("sqlite:///{db_path}")
        self.llm = WrappedQwenLLM(llm)
        self.sql_chain = SQLDatabaseChain.from_llm(llm=self.llm, db=self.db, verbose=False)
        
    def call(self, params: str, **kwargs) -> str:
        """调用工具
        
        Args:
            params: 工具参数，JSON字符串
            
        Returns:
            抽取结果，JSON字符串
        """
        # 解析参数
        # print("\n\n==== TOOL CALLED! ====")
        try:
            params_dict = json.loads(params)
            question = params_dict.get("question", "")

        except Exception as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"})


        try:
            result = sql_chain.run(question)
            return json.dumps({"answer": result}, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"error": e})
    

