import json
import re
from pathlib import Path
from typing import Dict, Optional
from langchain_core.prompts import PromptTemplate

class PromptLoader:
    """提示词模板加载器 (基于 LangChain PromptTemplate + 变量声明驱动渲染)"""
    
    def __init__(self, prompt_dir: str, global_variables: Optional[Dict[str, str]] = None):
        self.prompt_dir = Path(prompt_dir)
        assert self.prompt_dir.exists(), f"Prompt目录不存在: {self.prompt_dir}"
        self.global_variables = global_variables or {}
    
    def load_prompt(self, prompt_id: str) -> Dict:
        """根据ID加载Prompt"""
        prompt_path = self.prompt_dir / f"{prompt_id}.json"
        assert prompt_path.exists(), f"Prompt文件不存在: {prompt_path}"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        return prompt_data
    
    def render_prompt(self, prompt_id: str, variables: Dict[str, str]) -> str:
        """加载并渲染Prompt"""
        prompt_data = self.load_prompt(prompt_id)
        template_str = prompt_data['template']
        
        # 合并全局变量 + 局部变量
        full_vars = {**self.global_variables, **variables}
        
        # 只提取 prompt 声明需要的变量 → 形成 filtered_vars
        required_vars = [var['name'] for var in prompt_data.get('variables', [])]
        missing_vars = set(required_vars) - full_vars.keys()
        if missing_vars:
            raise ValueError(f"缺少变量: {missing_vars}，prompt_id={prompt_id}")
        
        # 提取只需要的变量
        filtered_vars = {var_name: full_vars[var_name] for var_name in required_vars}
        
        # 对需要转义的变量值（通常是描述型长文本）做 escape → 保证 JSON block 不出错
        # 可以定义规则：凡是变量名里含有 description / _text / _list 这种 → 自动转义
        for var_name, var_value in filtered_vars.items():
            if isinstance(var_value, str) and self._should_escape_variable(var_name):
                filtered_vars[var_name] = self._escape_braces(var_value)
        
        # 再处理模板中 ```json block``` 里面的 { } → 避免被 format 破坏
        template_str_safe = self._escape_braces_in_json_block(template_str)
        
        # 用 LangChain PromptTemplate 渲染
        prompt_template = PromptTemplate.from_template(template_str_safe)
        
        # Debug 输出
        # print(f"[CHECK][{prompt_id}] filtered_vars:", filtered_vars)
        
        rendered_prompt = prompt_template.format(**filtered_vars)
        return rendered_prompt
    
    @staticmethod
    def _escape_braces_in_json_block(text: str) -> str:
        """转义 JSON block 内的 { 和 }，避免 format 破坏"""
        def replacer(match):
            content = match.group(1)
            content_escaped = content.replace("{", "{{").replace("}", "}}")
            return f"```json\n{content_escaped}\n```"
        
        new_text = re.sub(r"```json\n(.*?)\n```", replacer, text, flags=re.DOTALL)
        return new_text
    
    @staticmethod
    def _should_escape_variable(var_name: str) -> bool:
        """判断变量是否需要转义 {}，防止 format 破坏"""
        keywords = ['description', '_text', '_list']
        return any(keyword in var_name for keyword in keywords)
    
    @staticmethod
    def _escape_braces(text: str) -> str:
        """对变量值里的 { 和 } 做转义"""
        return text.replace("{", "{{").replace("}", "}}")
