import json
import re
from typing import Dict, List, Any

try:
    from json_repair import repair_json as _json_repair
except Exception:
    _json_repair = None

DOC_TYPE_META: Dict[str, Dict[str, str]] = {
    "screenplay": {
        "section_label": "Scene",
        "title": "scene_name",
        "subtitle": "sub_scene_name",
        "contains_pred": "SCENE_CONTAINS",
    },
    "novel": {
        "section_label": "Chapter",
        "title": "chapter_name",
        "subtitle": "sub_chapter_name",
        "contains_pred": "CHAPTER_CONTAINS",
    },
     "general": {
        "section_label": "Document",
        "title": "title",
        "subtitle": "subtitle",
        "contains_pred": "CONTAINS",
    },
}

DOC_TYPE_DESCRIPTION: Dict[str, Dict[str, str]] = {
    "screenplay": {
        "entity": "- Scene: 场景，表示第几场戏，包含场次（scene_id）、场景名称（scene_name）、子场景名称（sub_scene_name）等信息。",
        "relation": "- SCENE_CONTAINS: 场景中包含关系（Scene → Any）",
    },
    "novel": {
        "entity": "- Chapter: 章节，表示小说的章节信息，包含章节标题（chapter_name）、子章节标题（sub_chapter_name）等信息。",
        "relation": "- CHAPTER_CONTAINS: 章节中包含关系（Chapter → Any）",
    },
    "general": {
        "entity": "- Document: 文档，表示文档的基本信息，包含标题（title）、副标题（subtitle）等信息。",
        "relation": "- CONTAINS: 文档中包含关系（Document → Any）",
    },
}



def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

import json
import re
from typing import Any, Tuple


# 匹配一行形如:   "key": value,
# 注意：我们只处理“单行字段”，因为跨行字符串无法安全修复
_LINE_KV_RE = re.compile(
    r'^(?P<indent>\s*)"(?P<key>[^"\\]+)"\s*:\s*(?P<val>.*?)(?P<tail>,?\s*)$'
)

# 允许的 JSON 原生字面量（不该被加引号）
_JSON_LITERALS = {"true", "false", "null"}

# 纯数字：整数/小数/科学计数
_NUM_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')


def _escape_json_string(s: str) -> str:
    # 最小必要转义：反斜杠和双引号
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


def _looks_like_unquoted_string(val: str) -> bool:
    """
    Decide whether a value token on a single line is likely an unquoted string.

    We DO NOT quote if it looks like:
    - already quoted string: "..."
    - object/array start: { ... or [ ...
    - JSON literal: true/false/null
    - number
    """
    v = val.strip()
    if v == "":
        return False

    # already quoted
    if v.startswith('"'):
        return False

    # object / array
    if v.startswith("{") or v.startswith("["):
        return False

    # json literals
    low = v.lower()
    if low in _JSON_LITERALS:
        return False

    # number
    if _NUM_RE.match(v):
        return False

    # otherwise treat as unquoted string
    return True



def fix_json_str(json_str: str, verbose: bool = False) -> str:
    """
    修复常见的 JSON 格式错误并返回修复后的 JSON 字符串
    
    核心策略:
    1. 找到所有 "key": value 模式
    2. 对于字符串值,移除双引号后重新添加
    3. 保留文本中的单引号(撇号,如 Sanji's)
    4. 保留已经正确转义的引号
    5. 修复其他常见问题(逗号、布尔值等)
    
    注意: 如果修复失败,函数会返回原始的 json_str 而不是抛出异常
    
    参数:
        json_str: 需要修复的 JSON 字符串
        verbose: 是否打印修复过程的详细信息
        
    返回:
        修复后的 JSON 字符串,如果修复失败则返回原始字符串
    """
    # 保存原始字符串
    original_json_str = json_str
    
    try:
        if verbose:
            print("开始修复 JSON 字符串...")
        
        # 步骤 1: 移除 BOM 标记
        if json_str.startswith('\ufeff'):
            json_str = json_str[1:]
            if verbose:
                print("- 移除了 BOM 标记")
        
        # 步骤 2: 转换 Python 风格的布尔值和 None
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        if verbose:
            print("- 转换了 Python 风格的布尔值和 None")
        
        # 步骤 2.5: 将 key 的单引号转换为双引号
        # 匹配 'key': 模式并转换为 "key":
        json_str = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)
        if verbose:
            print("- 转换了 key 的单引号为双引号")
        
        # 步骤 3: 处理所有的 "key": value 对
        # 策略: 找到每个 key,提取其值,判断类型并重新格式化
        
        def fix_key_value_pairs(text):
            """修复所有的 key-value 对"""
            lines = text.split('\n')
            result_lines = []
            fixed_count = 0
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # 匹配 "key": 模式
                match = re.match(r'^(\s*)"([^"]+)":\s*(.*)$', line)
                if match:
                    indent = match.group(1)
                    key = match.group(2)
                    value_part = match.group(3).strip()
                    
                    # 判断值的类型
                    # 如果以 { 或 [ 开头,是对象或数组,保持不变
                    if value_part.startswith('{') or value_part.startswith('['):
                        result_lines.append(line)
                        i += 1
                        continue
                    
                    # 如果是数字、布尔值或 null,保持不变
                    if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?,?$', value_part) or \
                       value_part.startswith(('true', 'false', 'null')):
                        result_lines.append(line)
                        i += 1
                        continue
                    
                    # 如果值以 " 开头并且包含正确的结束引号,检查是否是有效的字符串
                    if value_part.startswith('"'):
                        # 尝试找到匹配的结束引号
                        in_escape = False
                        quote_pos = -1
                        for idx in range(1, len(value_part)):
                            if in_escape:
                                in_escape = False
                                continue
                            if value_part[idx] == '\\':
                                in_escape = True
                                continue
                            if value_part[idx] == '"':
                                quote_pos = idx
                                break
                        
                        # 如果找到了匹配的引号,说明这是一个有效的字符串
                        if quote_pos > 0:
                            # 检查引号后面是否只有逗号或空格
                            after_quote = value_part[quote_pos+1:].strip()
                            if not after_quote or after_quote == ',':
                                result_lines.append(line)
                                i += 1
                                continue
                    
                    # 否则是字符串值,需要提取并重新格式化
                    # 提取值:从当前位置开始,直到遇到下一个 key 或结束符号
                    value_content = []
                    current_line_value = value_part
                    
                    # 保存转义的引号
                    current_line_value = current_line_value.replace('\\"', '<<<ESCAPED_DOUBLE_QUOTE>>>')
                    current_line_value = current_line_value.replace("\\'", '<<<ESCAPED_SINGLE_QUOTE>>>')
                    
                    # 只移除双引号,保留单引号(因为单引号在 JSON 字符串中是合法的)
                    current_line_value = current_line_value.replace('"', '')
                    
                    # 恢复转义的引号
                    current_line_value = current_line_value.replace('<<<ESCAPED_DOUBLE_QUOTE>>>', '"')
                    current_line_value = current_line_value.replace('<<<ESCAPED_SINGLE_QUOTE>>>', "'")
                    
                    # 移除尾随逗号
                    if current_line_value.endswith(','):
                        has_comma = True
                        current_line_value = current_line_value[:-1].strip()
                    else:
                        has_comma = False
                    
                    value_content.append(current_line_value)
                    
                    # 检查下一行是否是值的延续
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # 如果下一行以 " 开头(新的 key)或以 } ] 开头(结束),则值结束
                        if next_line.startswith('"') or next_line.startswith('}') or next_line.startswith(']'):
                            break
                        # 如果下一行为空,跳过
                        if not next_line:
                            j += 1
                            continue
                        # 否则是值的延续
                        next_line_clean = next_line
                        
                        # 保存转义的引号
                        next_line_clean = next_line_clean.replace('\\"', '<<<ESCAPED_DOUBLE_QUOTE>>>')
                        next_line_clean = next_line_clean.replace("\\'", '<<<ESCAPED_SINGLE_QUOTE>>>')
                        
                        # 只移除双引号
                        next_line_clean = next_line_clean.replace('"', '')
                        
                        # 恢复转义的引号
                        next_line_clean = next_line_clean.replace('<<<ESCAPED_DOUBLE_QUOTE>>>', '"')
                        next_line_clean = next_line_clean.replace('<<<ESCAPED_SINGLE_QUOTE>>>', "'")
                        
                        if next_line_clean.endswith(','):
                            has_comma = True
                            next_line_clean = next_line_clean[:-1].strip()
                        value_content.append(next_line_clean)
                        j += 1
                    
                    # 合并值内容
                    full_value = ' '.join(value_content).strip()
                    
                    # 转义内部的双引号和反斜杠
                    # 先转义反斜杠,再转义双引号
                    full_value = full_value.replace('\\', '\\\\')
                    full_value = full_value.replace('"', '\\"')
                    
                    # 检查下一行是否需要逗号
                    needs_comma = has_comma
                    if not needs_comma and j < len(lines):
                        next_line = lines[j].strip()
                        if next_line and next_line[0] == '"':
                            needs_comma = True
                    
                    # 重新格式化为正确的 JSON
                    if needs_comma:
                        result_lines.append(f'{indent}"{key}": "{full_value}",')
                    else:
                        result_lines.append(f'{indent}"{key}": "{full_value}"')
                    
                    fixed_count += 1
                    i = j
                else:
                    result_lines.append(line)
                    i += 1
            
            return '\n'.join(result_lines), fixed_count
        
        json_str, fixed_count = fix_key_value_pairs(json_str)
        if verbose and fixed_count > 0:
            print(f"- 修复了 {fixed_count} 个 key-value 对")
        
        # 步骤 4: 移除对象或数组最后一个元素后的多余逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        if verbose:
            print("- 移除了多余的尾随逗号")
        
        # 步骤 5: 添加缺少的逗号(针对数组元素)
        # 匹配 } 或 ] 后直接跟 { 或 [ 的情况
        json_str = re.sub(r'\}(\s*)\n(\s*)\{', r'},\1\n\2{', json_str)
        json_str = re.sub(r'\](\s*)\n(\s*)\[', r'],\1\n\2[', json_str)
        if verbose:
            print("- 添加了缺少的逗号")
        
        # 步骤 6: 再次移除可能产生的多余逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 步骤 7: 验证修复后的 JSON
        json.loads(json_str)
        if verbose:
            print("✓ JSON 修复成功!")
        return json_str
        
    except Exception as e:
        if verbose:
            print(f"✗ JSON 修复失败: {e}")
            print("✗ 返回原始 JSON 字符串")
        return original_json_str


def fix_and_parse_json(json_str: str, verbose: bool = False) -> Any:
    """
    修复 JSON 字符串并解析为 Python 对象
    
    注意: 如果修复失败,此函数会抛出异常
    
    参数:
        json_str: 需要修复的 JSON 字符串
        verbose: 是否打印修复过程的详细信息
        
    返回:
        解析后的 Python 对象(dict, list, 等)
        
    异常:
        ValueError: 如果无法修复或解析 JSON 字符串
    """
    fixed_json = fix_json_str(json_str, verbose=verbose)
    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        error_msg = f"无法解析 JSON 字符串。错误: {e.msg} (行 {e.lineno}, 列 {e.colno})"
        if verbose:
            print(f"✗ {error_msg}")
        raise ValueError(error_msg)


def extract_first_json_obj(text: str) -> str:
    """
    Extract first complete JSON object/array from noisy text.
    """
    if not text or not isinstance(text, str):
        return ""

    s = text.strip()

    # find first { or [
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return ""

    stack = []
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                continue
            left = stack.pop()
            if (left == "{" and ch != "}") or (left == "[" and ch != "]"):
                return s[start : j + 1].strip()
            if not stack:
                return s[start : j + 1].strip()

    return s[start:].strip()



def _extract_json_code(text: str) -> str:
    """提取 ```json ... ``` 内部内容；若没有 code block 就原样返回"""
    text = remove_think_tags(text)
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)

    return (m.group(1).strip() if m else text.strip().strip("`").strip())


def extract_first_json_block(text: str) -> str:
    """
    提取 text 中第一个完整 JSON object/array（{...} 或 [...]），并截断尾部垃圾。
    找不到就返回 ""。
    """
    if not text:
        return ""
    s = text.strip()

    # 找到第一个 { 或 [
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return ""

    stack = []
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                # 多余的闭括号，忽略
                continue
            left = stack.pop()
            if (left == "{" and ch != "}") or (left == "[" and ch != "]"):
                # 括号不匹配，直接返回当前最好的前缀
                return s[start:j+1].strip()
            if not stack:
                return s[start:j+1].strip()

    # 没闭合完整
    return s[start:].strip()


def _escape_inner_quotes(json_txt: str) -> str:
    """
    逐字符扫描：在 JSON **字符串 value 内**，将裸引号 " → \"
    规则：位于字符串内部且其后首个非空白字符不是 , } ] : 之一，就视作“内部引号”
    """
    out, in_str, esc = [], False, False
    for i, ch in enumerate(json_txt):
        if ch == '"' and not esc:
            if in_str:  # 目前在字符串里
                # 预览下一个非空白字符，判断它是不是结束符
                j = i + 1
                while j < len(json_txt) and json_txt[j] in " \t\r\n":
                    j += 1
                if j < len(json_txt) and json_txt[j] not in ",}]:":
                    # 不是结束，引号需要转义
                    out.append(r'\"')
                    continue
                else:      # 正常结束
                    in_str = False
                    out.append(ch)
                # toggle 完毕
            else:          # 进入字符串
                in_str = True
                out.append(ch)
        else:
            out.append(ch)

        # 处理反斜杠转义状态
        esc = (ch == '\\') and not esc if in_str else False

    return "".join(out)


def _ensure_matching_outer_brackets(text: str) -> str:
    """
    Ensure the outer JSON boundary pair matches:
    - starts with '{' -> ends with '}'
    - starts with '[' -> ends with ']'
    """
    if not text:
        return text

    s = text.strip()
    if not s:
        return text

    # Keep only content from the first JSON opener.
    opener_idx = None
    for i, ch in enumerate(s):
        if ch in "{[":
            opener_idx = i
            break
    if opener_idx is None:
        return text
    s = s[opener_idx:].strip()
    if not s:
        return text

    opener = s[0]
    if opener not in "{[":
        return s

    expected_closer = "}" if opener == "{" else "]"
    tail_idx = len(s) - 1
    while tail_idx >= 0 and s[tail_idx].isspace():
        tail_idx -= 1
    if tail_idx < 0:
        return s + expected_closer

    tail = s[tail_idx]
    if tail in "}]":
        if tail != expected_closer:
            s = s[:tail_idx] + expected_closer + s[tail_idx + 1 :]
        return s

    return s + expected_closer


def _repair_with_json_repair(text: str) -> str:
    if _json_repair is None:
        return ""
    try:
        repaired = _json_repair(
            text,
            return_objects=False,
            skip_json_loads=True,
            ensure_ascii=False,
        )
    except TypeError:
        try:
            repaired = _json_repair(text)
        except Exception:
            return ""
    except Exception:
        return ""
    return repaired if isinstance(repaired, str) else ""


def correct_json_format(text: str) -> str:
    body0 = _extract_json_code(text)
    body0 = extract_first_json_block(body0) or body0
    body0 = _ensure_matching_outer_brackets(body0)
    if is_valid_json(body0):
        return body0

    body = body0
    body = body.replace("True", "true").replace("False", "false")
    body = _escape_inner_quotes(body)
    body = _ensure_matching_outer_brackets(body)

    if is_valid_json(body):
        return body

    patched = patch_chinese_quotes(body)
    patched = _ensure_matching_outer_brackets(patched)
    if is_valid_json(patched):
        return patched

    repaired = _repair_with_json_repair(body)
    repaired = extract_first_json_block(repaired) or repaired
    repaired = _ensure_matching_outer_brackets(repaired)
    if is_valid_json(repaired):
        return repaired

    final_fix = fix_json_str(text, verbose=False)
    if is_valid_json(final_fix):
        return final_fix

    return text


def patch_chinese_quotes(json_str: str) -> str:
    RIGHT_FANCY_QUOTES = "”»›」』‟"
    
    # === 1. 修复中文右引号后缺英文引号 ===
    _FANCY_CLOSE_RE = re.compile(
        rf'(?P<quote>[{RIGHT_FANCY_QUOTES}])(?P<tail>\s*(?:,|\]|}}|:))'
    )
    patched = _FANCY_CLOSE_RE.sub(r'\g<quote>"\g<tail>', json_str)

    # === 2. 修复最后一个非 ] 或 } 的字符不是 " 的情况 ===
    # 找倒数第一个不是空格、换行、]、} 的字符
    tail_match = re.search(r'[^ \n\[\r\t\]\}](?=[ \n\r\t\]\}]*$)', patched)
    if tail_match:
        char = tail_match.group()
        if char != '"':
            insert_pos = tail_match.end()
            patched = patched[:insert_pos] + '"' + patched[insert_pos:]

    return patched



def normalize_quotes(txt: str) -> str:
    """把中文 / 花体引号全部替换为标准双引号"""
    RE_FANCY_QUOTES = re.compile(r"[“”„‟«»「」『』‹›]")
    return RE_FANCY_QUOTES.sub('"', txt)

def safe_text_for_json(raw: str) -> str:
    """
    处理一下输入长文中中的中文引号问题。
    """
    txt = raw.lstrip('\ufeff')                      # 1) 去 BOM
    txt = txt.replace('\r\n', '\n').replace('\r', '\n').strip()
    txt = re.sub(r'[ \t]{2,}', ' ', txt)           # 2) 压缩空白
    txt = normalize_quotes(txt)                    # 3) 引号标准化
    return json.dumps(txt, ensure_ascii=False)[1:-1]

def is_valid_json(text: str) -> bool:
    try:
        # body = _extract_json_code(text)
        # body = extract_first_json_block(body) or body
        json.loads(text)
        return True
    except Exception:
        return False
