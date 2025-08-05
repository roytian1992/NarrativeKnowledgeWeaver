# kag/utils/tools_loader.py

from typing import List, Any

def load_all_tools(prompt_loader, llm) -> List[Any]:
    """加载全部 Tools
    
    根据LLM类型加载不同的工具实现
    - 对于OpenAI，使用LangChain的Tool
    - 对于Qwen3，使用Qwen-Agent的BaseTool
    
    Args:
        prompt_loader: 提示词加载器
        llm: 语言模型
        
    Returns:
        工具列表
    """
    # 判断LLM类型
    llm_type = getattr(llm, "_llm_type", "")


    if llm_type == "qwen_openai_fc" or hasattr(llm, "tokenizer"):
        # Qwen3模式：使用Qwen-Agent的BaseTool
        print("使用Qwen-Agent工具")
        from kag.tools.qwen_tools import (
            QwenEntityExtractionTool,
            QwenRelationExtractionTool,
            QwenSceneElementsExtractionTool,
            QwenExtractPropsTool,
            QwenReflectExtractionTool,
        )
        
        tools = [
            QwenEntityExtractionTool(prompt_loader, llm),
            QwenRelationExtractionTool(prompt_loader, llm),
            QwenSceneElementsExtractionTool(prompt_loader, llm),
            QwenExtractPropsTool(prompt_loader, llm),
            QwenReflectExtractionTool(prompt_loader, llm),
        ]
    else:
        # OpenAI模式：使用LangChain的Tool
        print("使用LangChain工具")
        from kag.tools.entity_extraction_tool import build_entity_extraction_tool
        from kag.tools.relation_extraction_tool import build_relation_extraction_tool
        from kag.tools.scene_elements_extraction_tool import build_scene_elements_extraction_tool
        from kag.tools.extract_props_tool import build_extract_props_tool
        from kag.tools.reflect_extraction_tool import build_reflect_extraction_tool
        
        tools = [
            build_entity_extraction_tool(prompt_loader, llm),
            build_relation_extraction_tool(prompt_loader, llm),
            build_script_elements_extraction_tool(prompt_loader, llm),
            build_parse_scene_name_tool(prompt_loader, llm),
            build_extract_props_tool(prompt_loader, llm),
            build_reflect_extraction_tool(prompt_loader, llm),
        ]
    
    return tools
