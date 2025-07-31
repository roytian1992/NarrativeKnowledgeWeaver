"""
集成format.py的使用示例
展示如何在现有function中使用增强的JSON处理工具
"""
from typing import Dict, Any, List
import json
import logging
from kag.utils.function_manager import EnhancedJSONUtils, process_with_format_guarantee

logger = logging.getLogger(__name__)


repair_template = """
请修复以下实体提取结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含：
1. "entities"字段，且为非空数组
2. 每个实体包含"name"和"type"字段
3. JSON格式正确

请直接返回修复后的JSON，不要包含解释。
"""


class EntityExtractor:
    """
    集成format.py的实体提取器
    确保最终返回的是correct_json_format处理后的结果
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = ["entities"]
        self.field_validators = {
            "entities": lambda x: isinstance(x, list) and len(x) > 0
        }
        
        # 修复提示词模板
        self.repair_template = repair_template
    
    def call(self, params: str, **kwargs) -> str:
        """
        调用实体提取，保证返回correct_json_format处理后的结果
        
        Args:
            params: 参数字符串
            **kwargs: 其他参数
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        try:
            # 解析参数
            params_dict = json.loads(params)
            text = params_dict.get("text", "")
            entity_type_description_text = params_dict.get("entity_type_description_text", "")
            abbreviations = params_dict.get("abbreviations", "")
            reflection_results = params_dict.get("reflection_results", {})
            
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            # 即使是错误结果，也要经过correct_json_format处理
            error_result = {"error": f"参数解析失败: {str(e)}", "entities": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        if not text:
            error_result = {"error": "缺少文本内容", "entities": []}
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))
        
        try:
            # 构建提示词变量
            variables = {
                'text': text,
                'entity_type_description_text': entity_type_description_text,
                'abbreviations': abbreviations
            }
            
            # 渲染提示词
            prompt_text = self.prompt_loader.render_prompt('extract_entities_prompt', variables)
            
            # 构建消息
            messages = []
            
            # 添加反思信息作为背景
            previous_issues = reflection_results.get("issues", [])
            previous_suggestions = reflection_results.get("suggestions", [])
            
            background_info = ""
            if previous_suggestions:
                previous_issues_str = "\n".join(previous_issues)
                background_info += f"在进行实体提取的过程中，请特别关注：\n{previous_issues_str}\n相关问题的建议：\n{previous_suggestions}\n"
            
            if background_info:
                messages.append({"role": "user", "content": background_info})
            
            messages.append({"role": "user", "content": prompt_text})
            
            # 使用增强工具处理响应，保证返回correct_json_format处理后的结果
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            
            logger.info("实体提取完成，返回格式化后的JSON")
            return corrected_json
            
        except Exception as e:
            logger.error(f"实体提取过程中出现异常: {e}")
            error_result = {
                "error": f"实体提取失败: {str(e)}",
                "entities": []
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

class PlotGeneratorWithFormat:
    """
    集成format.py的情节生成器
    """
    
    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm
        
        # 定义验证规则
        self.required_fields = [
            "plot_unit_id", "title", "description", "theme", 
            "conflict", "participants"
        ]
        
        self.field_validators = {
            "title": lambda x: isinstance(x, str) and len(x.strip()) > 0,
            "description": lambda x: isinstance(x, str) and len(x.strip()) > 10,
            "participants": lambda x: isinstance(x, list) and len(x) > 0
        }
        
        self.repair_template = """
请修复以下情节单元生成结果中的问题：

原始响应：{original_response}
错误信息：{error_message}

请确保返回的JSON包含所有必需字段并且格式正确。
请直接返回修复后的完整JSON，不要包含解释。
"""
    
    def generate_plot_description(self, 
                                events: List[Dict], 
                                cluster_info: Dict, 
                                causality_threshold: str = "Medium") -> str:
        """
        生成情节描述，返回correct_json_format处理后的JSON字符串
        
        Args:
            events: 事件列表
            cluster_info: 聚类信息
            causality_threshold: 因果关系阈值
            
        Returns:
            str: 经过correct_json_format处理的JSON字符串
        """
        logger.info(f"开始生成情节描述，事件数量: {len(events)}")
        
        try:
            # 准备提示词变量
            variables = {
                "events": json.dumps(events, ensure_ascii=False, indent=2),
                "cluster_info": json.dumps(cluster_info, ensure_ascii=False, indent=2),
                "causality_threshold": causality_threshold
            }
            
            # 渲染提示词
            prompt = self.prompt_loader.render_prompt("plot_unit_construction_prompt", variables)
            
            # 构建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 使用增强工具处理响应
            corrected_json = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template
            )
            
            logger.info("情节描述生成完成")
            return corrected_json
            
        except Exception as e:
            logger.error(f"生成情节描述时出错: {str(e)}")
            error_result = {
                "plot_unit_id": f"plot_exception_{cluster_info.get('cluster_id', 'unknown')}",
                "title": "异常情节单元",
                "description": f"生成过程中出现异常: {str(e)}",
                "theme": "异常处理",
                "conflict": "系统异常",
                "participants": [],
                "error": str(e)
            }
            from kag.utils.format import correct_json_format
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

# 使用示例：展示如何简化现有代码
def show_integration_example():
    """展示集成format.py的使用示例"""
    
    print("=== 原来的复杂逻辑 ===")
    print("""
    # 原来需要手动处理格式和重试
    full_response = ""
    max_round = 3
    
    for i in range(max_round):
        result = self.llm.run(messages, enable_thinking=(i==0))
        content = result[0]['content']
        full_response += content.strip()
        
        # 手动格式化
        corrected = correct_json_format(full_response)
        if is_valid_json(corrected):
            return corrected
        
        messages.append({"role": "user", "content": "请修复JSON..."})
    
    # 最后的修复尝试...
    """)
    
    print("\n=== 现在的简洁逻辑 ===")
    print("""
    # 现在只需要一行调用，自动保证返回correct_json_format处理后的结果
    corrected_json = process_with_format_guarantee(
        llm_client=self.llm,
        messages=messages,
        required_fields=self.required_fields,
        field_validators=self.field_validators,
        repair_template=self.repair_template
    )
    return corrected_json  # 已经是correct_json_format处理后的结果
    """)

def test_format_integration():
    """测试format.py集成"""
    from kag.utils.enhanced_json_utils import get_corrected_json, is_valid_json_enhanced, analyze_json_issues
    
    # 测试数据
    test_cases = [
        '{"entities": [{"name": "测试", "type": "实体"}]}',  # 正常JSON
        '{"entities": [{"name": "测试", "type": "实体"}]',   # 缺少结束括号
        '```json\n{"entities": []}\n```',                  # 代码块格式
        '{"entities": [{"name": "测试中的"引号", "type": "实体"}]}',  # 内部引号问题
        '',  # 空响应
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"原始: {test_case}")
        
        # 获取格式化后的结果
        corrected = get_corrected_json(test_case)
        print(f"格式化后: {corrected}")
        
        # 增强验证
        is_valid = is_valid_json_enhanced(test_case, required_fields=["entities"])
        print(f"验证结果: {is_valid}")
        
        # 问题分析
        issues = analyze_json_issues(test_case, required_fields=["entities"])
        print(f"问题分析: {issues}")

if __name__ == "__main__":
    show_integration_example()
    print("\n" + "="*50)
    test_format_integration()

