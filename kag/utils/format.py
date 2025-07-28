import json
import re


def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
    return body



def is_valid_json(text: str) -> bool:
    try:
        # 尝试从 JSON 起始位置截取解析
        content = correct_json_format(text)
        json.loads(content)
        return True
    except Exception:
        return False