"""
增强的JSON处理工具
基于现有的format.py，添加智能重试和问题诊断功能
"""
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from kag.utils.format import correct_json_format, is_valid_json

logger = logging.getLogger(__name__)

class JSONIssueType(Enum):
    """JSON问题类型"""
    VALID = "valid"
    INVALID_AFTER_CORRECTION = "invalid_after_correction"  # 经过correct_json_format后仍无效
    MISSING_REQUIRED_FIELDS = "missing_required_fields"    # 缺少必需字段
    INVALID_FIELD_VALUES = "invalid_field_values"          # 字段值无效
    EMPTY_RESPONSE = "empty_response"                      # 空响应

class EnhancedJSONUtils:
    """增强的JSON处理工具类"""
    
    @staticmethod
    def analyze_json_response(content: str, 
                            required_fields: Optional[List[str]] = None,
                            field_validators: Optional[Dict[str, Callable]] = None) -> Tuple[JSONIssueType, str, Optional[Dict]]:
        """
        分析JSON响应，判断问题类型
        
        Args:
            content: 响应内容
            required_fields: 必需字段列表
            field_validators: 字段验证器字典
            
        Returns:
            Tuple: (问题类型, 错误信息, 解析结果或None)
        """
        if not content or content.strip() == "":
            return JSONIssueType.EMPTY_RESPONSE, "响应为空", None
        
        # 使用现有的correct_json_format处理
        corrected_content = correct_json_format(content)
        
        # 使用现有的is_valid_json检查
        if not is_valid_json(corrected_content):
            return JSONIssueType.INVALID_AFTER_CORRECTION, "经过格式修正后仍无法解析为有效JSON", None
        
        # 解析JSON
        try:
            parsed = json.loads(corrected_content)
        except json.JSONDecodeError as e:
            return JSONIssueType.INVALID_AFTER_CORRECTION, f"JSON解析失败: {e}", None
        
        # 检查必需字段
        if required_fields:
            missing_fields = [field for field in required_fields if field not in parsed]
            if missing_fields:
                return JSONIssueType.MISSING_REQUIRED_FIELDS, f"缺少必需字段: {missing_fields}", parsed
        
        # 检查字段值有效性
        if field_validators:
            for field, validator in field_validators.items():
                if field in parsed:
                    try:
                        if not validator(parsed[field]):
                            return JSONIssueType.INVALID_FIELD_VALUES, f"字段 {field} 验证失败", parsed
                    except Exception as e:
                        return JSONIssueType.INVALID_FIELD_VALUES, f"字段 {field} 验证异常: {e}", parsed
        
        return JSONIssueType.VALID, "JSON有效", parsed
    
    @staticmethod
    def enhanced_json_validation(content: str, 
                               required_fields: Optional[List[str]] = None,
                               field_validators: Optional[Dict[str, Callable]] = None) -> Tuple[bool, str, Optional[Dict], str]:
        """
        增强的JSON验证，返回处理后的内容
        
        Args:
            content: JSON内容
            required_fields: 必需字段列表
            field_validators: 字段验证器字典
            
        Returns:
            Tuple: (是否有效, 错误信息, 解析结果或None, 处理后的JSON字符串)
        """
        # 先用现有函数处理格式
        corrected_content = correct_json_format(content)
        
        # 分析问题
        issue_type, error_msg, parsed = EnhancedJSONUtils.analyze_json_response(
            content, required_fields, field_validators
        )
        # print("[CHECK]: ", issue_type, error_msg, parsed)
        is_valid = (issue_type == JSONIssueType.VALID)
        return is_valid, error_msg, parsed, corrected_content
    
    @staticmethod
    def process_llm_response_with_retry(llm_client,
                                      initial_messages: List[Dict],
                                      required_fields: Optional[List[str]] = None,
                                      field_validators: Optional[Dict[str, Callable]] = None,
                                      max_retries: int = 3,
                                      enable_thinking: bool = True,
                                      repair_prompt_template: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        处理LLM响应，包含重试和修复机制，确保返回correct_json_format处理后的结果
        
        Args:
            llm_client: LLM客户端
            initial_messages: 初始消息
            required_fields: 必需字段
            field_validators: 字段验证器
            max_retries: 最大重试次数
            enable_thinking: 是否启用思考
            repair_prompt_template: 修复提示词模板
            
        Returns:
            Tuple: (处理结果字典, 最终的JSON字符串)
        """
        logger.info("开始处理LLM响应")
        
        # 第一次调用
        result = llm_client.run(initial_messages, enable_thinking=enable_thinking)
        content = result[0]['content'] if isinstance(result, list) else result.get('content', '')
        
        # 验证响应
        is_valid, error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
            content, required_fields, field_validators
        )
        
        if is_valid:
            logger.info("响应有效，直接返回")
            return parsed, corrected_content
        
        # 尝试重试和修复
        current_content = content
        for attempt in range(max_retries):
            print(f"尝试修复响应，第 {attempt + 1} 次")
            print("[CHECK] current_content: ", current_content)
            
            try:
                # 如果有修复提示词模板，使用LLM修复
                if repair_prompt_template:
                    repair_messages = initial_messages.copy()
                    repair_prompt = repair_prompt_template.format(
                        original_response=current_content,
                        error_message=error_msg
                    )
                    repair_messages.append({"role": "user", "content": repair_prompt})
                    
                    # 调用LLM修复
                    repair_result = llm_client.run(repair_messages, enable_thinking=False)
                    repair_content = repair_result[0]['content'] if isinstance(repair_result, list) else repair_result.get('content', '')
                    
                    # 验证修复结果
                    is_valid, new_error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
                        repair_content, required_fields, field_validators
                    )
                    
                    if is_valid:
                        logger.info(f"LLM修复成功，第 {attempt + 1} 次尝试")
                        return parsed, corrected_content
                    
                    # 更新内容用于下次尝试
                    current_content = repair_content
                    error_msg = new_error_msg
                else:
                    # 没有修复模板，使用通用修复提示
                    repair_messages = initial_messages.copy()
                    repair_messages.append({
                        "role": "user",
                        "content": f"请修复以下JSON响应中的问题：\n问题：{error_msg}\n原始响应：{current_content}\n请返回修复后的完整JSON。"
                    })
                    
                    repair_result = llm_client.run(repair_messages, enable_thinking=enable_thinking)
                    repair_content = repair_result[0]['content'] if isinstance(repair_result, list) else repair_result.get('content', '')
                    
                    # 验证修复结果
                    is_valid, new_error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
                        repair_content, required_fields, field_validators
                    )
                    
                    if is_valid:
                        logger.info(f"通用修复成功，第 {attempt + 1} 次尝试")
                        return parsed, corrected_content
                    
                    current_content = repair_content
                    error_msg = new_error_msg
                
            except Exception as e:
                logger.warning(f"修复尝试 {attempt + 1} 失败: {e}")
                continue
        
        # 所有尝试都失败，返回错误结果，但仍使用correct_json_format处理
        logger.error("所有修复尝试都失败，最后一次整任务尝试")
        result = llm_client.run(initial_messages, enable_thinking=enable_thinking)
        content = result[0]['content'] if isinstance(result, list) else result.get('content', '')
        final_corrected = correct_json_format(content)
        
        print("[CHECK] 最终结果: ", final_corrected)
        error_result = {
            "error": "响应处理失败",
            "original_content": content,
            "error_details": error_msg,
            "attempts": max_retries + 1
        }
        
        return error_result, final_corrected

# 便捷函数，保持与现有代码的兼容性
def is_valid_json_enhanced(content: str, 
                          required_fields: Optional[List[str]] = None,
                          field_validators: Optional[Dict[str, Callable]] = None) -> bool:
    """增强版的is_valid_json函数，基于现有的format.py"""
    is_valid, _, _, _ = EnhancedJSONUtils.enhanced_json_validation(content, required_fields, field_validators)
    return is_valid

def get_corrected_json(content: str) -> str:
    """获取经过correct_json_format处理后的JSON字符串"""
    return correct_json_format(content)

def analyze_json_issues(content: str, 
                       required_fields: Optional[List[str]] = None,
                       field_validators: Optional[Dict[str, Callable]] = None) -> str:
    """分析JSON问题的便捷函数"""
    issue_type, error_msg, _ = EnhancedJSONUtils.analyze_json_response(content, required_fields, field_validators)
    return f"{issue_type.value}: {error_msg}"

def process_with_format_guarantee(llm_client,
                                messages: List[Dict],
                                required_fields: Optional[List[str]] = None,
                                field_validators: Optional[Dict[str, Callable]] = None,
                                max_retries: int = 3,
                                enable_thinking: bool = True,
                                repair_template: Optional[str] = None) -> str:
    """
    处理LLM响应并保证返回correct_json_format处理后的结果
    
    Returns:
        str: 经过correct_json_format处理的JSON字符串
    """
    result_dict, corrected_json = EnhancedJSONUtils.process_llm_response_with_retry(
        llm_client=llm_client,
        initial_messages=messages,
        required_fields=required_fields,
        field_validators=field_validators,
        max_retries=max_retries,
        enable_thinking=enable_thinking,
        repair_prompt_template=repair_template
    )
    status = result_dict.get("error", "")
    if status:
        status = "error"
    else:
        status = "success"
        
    return corrected_json, status

