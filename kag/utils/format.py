import json
import re


def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def simple_fix(raw: str) -> str:
    """
    仅做两种轻量修复：
      1. 行内单双引号不配对 => 在行尾补 "
      2. 对象 / 数组结尾忘写逗号 => 自动补 ,
    其余内容原样保留。
    """
    fixed_lines = []
    quote_pat = re.compile(r'(?<!\\)"')  # 匹配非转义 "
    
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        # 1) 补缺失的右引号
        if (quote_pat.findall(line) and len(quote_pat.findall(line)) % 2 == 1):
            line += '"'
        
        fixed_lines.append(line)
        
        # 2) 如果本行以 ] 或 } 结束，而下一行看起来是字面量且没有逗号 -> 补 ,
        if i < len(lines) - 1:
            stripped = line.rstrip()
            next_line = lines[i + 1].lstrip()
            if stripped.endswith((']', '}')) and not stripped.endswith((',', '],', '},')):
                if re.match(r'["{\[]', next_line):  # 下一行以 " { [ 之一开头
                    fixed_lines[-1] = stripped + ','

    return '\n'.join(fixed_lines)


def _extract_json_code(text: str) -> str:
    """提取 ```json ... ``` 内部内容；若没有 code block 就原样返回"""
    text = remove_think_tags(text)
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    return (m.group(1) if m else text.strip().strip("`"))

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

def correct_json_format(text: str) -> str:
    body = _extract_json_code(text)
    body = body.replace("True", "true").replace("False", "false")
    body = _escape_inner_quotes(body)
    if is_valid_json(body):
        return body
    elif is_valid_json(patch_chinese_quotes(body)):
        return patch_chinese_quotes(body)
    else:
        return simple_fix(body)


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
        # 尝试从 JSON 起始位置截取解析
        content = correct_json_format(text)
        json.loads(content)
        return True
    except Exception:
        return False