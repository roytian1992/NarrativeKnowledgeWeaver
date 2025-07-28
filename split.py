import fitz
import json
import re
from itertools import count

# ---------- 1. 工具函数 ----------
def get_alignment(pw, x0, x1, ltol=50, rtol=50, ctol=50):
    pc = pw / 2
    lc = (x0 + x1) / 2
    if x0 < ltol:
        return "Left"
    elif abs(lc - pc) < ctol:
        return "Center"
    elif (pw - x1) < rtol:
        return "Right"
    return "Normal"

def is_line_bold(line):
    total = bold = 0
    for sp in line.get("spans", []):
        txt = sp.get("text", "").strip()
        if not txt:
            continue
        total += len(txt)
        if "bold" in sp.get("font", "").lower():
            bold += len(txt)
    return total > 0 and bold / total >= .5

def is_page_number_or_blank(txt):
    s = txt.strip()
    return not s or re.fullmatch(r"\d+", s) is not None

RE_REMARK_LINE = re.compile(r"^[（(][^）)]+[）)](?:[（(][^）)]+[）)])*$")
def is_remark_line(txt):
    return bool(RE_REMARK_LINE.fullmatch(txt.strip()))

# ---------- 2. 行迭代 ----------
def iter_lines(doc):
    for pno, page in enumerate(doc):
        pw = page.rect.width
        for blk in page.get_text("dict")["blocks"]:
            if "lines" not in blk:
                continue
            for line in blk["lines"]:
                txt = "".join(sp["text"] for sp in line["spans"])
                if is_page_number_or_blank(txt):
                    continue
                x0 = min(sp["bbox"][0] for sp in line["spans"])
                x1 = max(sp["bbox"][2] for sp in line["spans"])
                yield {
                    "text": txt.rstrip("\n"),
                    "bold": is_line_bold(line),
                    "alignment": get_alignment(pw, x0, x1),
                    "page": pno + 1,
                }

# ---------- 3. 正则 ----------
MAIN_SCENE_RE = re.compile(r"^\d+[、\.]")
SUB_SCENE_RE  = re.compile(r"^\d+-[0-9A-Za-z]+[、\.]")
INLINE_SUB_RE = re.compile(r"\d+-[0-9A-Za-z]+[、\.]")
NAME_TYPE_RE  = re.compile(
    r"""^\s*
        (?P<name>.+?)\s*
        (?:[（(]\s*(?P<type>V\.?O\.?|O\.?S\.?)\s*[）)])?
        \s*$""", re.I | re.X
)

# ---------- 4. 帮手 ----------
def append_content(buf, tag, text):
    if tag == "[描述]" and buf and buf[-1].startswith("[描述]"):
        buf[-1] += text.lstrip()
    else:
        buf.append(f"{tag}{text}")

def merge_main_scene(lines, i):
    """
    合并跨行主场景标题：

    - 允许 Bold+Left 第一行；
    - 后续若仍是 Bold 且 Left/Center，并且 **不含括号 “（” “(”**（否则当作角色行），就继续合并；
    - 遇到下一行是主场景 / 子场景 / 角色行 / 普通行则停止。
    """
    parts, j = [lines[i]["text"].strip()], i + 1
    while j < len(lines):
        ln = lines[j]
        txt = ln["text"].strip()
        if (ln["bold"]
            and ln["alignment"] in ("Left", "Center")
            and "（" not in txt and "(" not in txt      # 关键：排除角色行
            and not MAIN_SCENE_RE.match(txt)
            and not SUB_SCENE_RE.match(txt)):
            parts.append(txt)
            j += 1
        else:
            break
    return " ".join(parts), j

def parse_dialogue(lines, i, counter):
    """
    返回 (dialog_obj | None, new_index)
    若没有正文 & 备注，则视作误判，返回 None。
    """
    raw = lines[i]["text"].strip()
    i += 1
    m = NAME_TYPE_RE.match(raw)
    char  = m.group("name").strip() if m else raw
    vtype = (m.group("type") or "").upper().replace(" ", "").replace(".", "")

    remarks = []
    while i < len(lines) and is_remark_line(lines[i]["text"]):
        remarks.extend(re.findall(r"[（(]([^）)]+)[）)]", lines[i]["text"]))
        i += 1

    speech = []
    while i < len(lines):
        ln = lines[i]
        if ln["bold"] or ln["alignment"] in ("Left", "Right") or is_remark_line(ln["text"]):
            break
        speech.append(ln["text"])
        i += 1

    speech_text = "".join(speech).strip()
    if not speech_text and not remarks:          # 误判
        return None, i

    return {
        "_id": next(counter),
        "character": char,
        "type": vtype,
        "remark": remarks,
        "content": speech_text
    }, i

# ---------- 5. 主解析 ----------
def split_script(doc):
    lines = list(iter_lines(doc))
    scenes = []
    scn_counter = count(1)
    dlg_counter = count(1)

    major, sub = None, ""
    buf, convs = [], []
    i = 0
    while i < len(lines):
        ln = lines[i]
        t = ln["text"].strip()

        # ---------- 主场景 ----------
        if ln["bold"] and ln["alignment"] == "Left" and MAIN_SCENE_RE.match(t):
            # 把上一个场景入库
            if major:
                scenes.append({
                    "_id": next(scn_counter),
                    "scene_name": major,
                    "sub_scene_name": sub,
                    "content": "\n".join(buf),
                    "conversation": convs
                })

            # 合并多行主场景标题
            major_line, i = merge_main_scene(lines, i)

            # 行内子场景
            m = INLINE_SUB_RE.search(major_line)
            if m:
                major = major_line[:m.start()].strip()
                sub   = major_line[m.start():].strip()
            else:
                major = major_line.strip()
                sub   = ""

            buf, convs = [], []
            continue

        # ---------- 子场景（独立行） ----------
        if ln["bold"] and SUB_SCENE_RE.match(t):
            if major:
                scenes.append({
                    "_id": next(scn_counter),
                    "scene_name": major,
                    "sub_scene_name": sub,
                    "content": "\n".join(buf),
                    "conversation": convs
                })
            sub, buf, convs = t, [], []
            i += 1
            continue

        # ---------- 对话 ----------
        if ln["bold"] and ln["alignment"] == "Center" and not SUB_SCENE_RE.match(t):
            dlg, i2 = parse_dialogue(lines, i, dlg_counter)
            if dlg:                                   # 真·对白
                convs.append(dlg)
                bracket_items = []
                if dlg["type"]:
                    bracket_items.append(dlg["type"])
                bracket_items.extend(dlg["remark"])
                bracket = f"（{'、'.join(bracket_items)}）" if bracket_items else ""
                append_content(buf, "[对话]", f"{dlg['character']}{bracket}：{dlg['content']}")
            else:                                     # 误判标题残片
                append_content(buf, "[描述]", t)
            i = i2
            continue

        # ---------- 舞台提示 ----------
        if ln["alignment"] == "Right" and not ln["bold"]:
            append_content(buf, "[舞台提示]", t)
            i += 1
            continue

        # ---------- 描述 / 其他 ----------
        if not ln["bold"]:
            tag = "[描述]" if (
                ln["alignment"] in ("Left", "Normal", "Center")
            ) else "[其他]"
            append_content(buf, tag, t)
        else:
            append_content(buf, "[其他]", t)
        i += 1

    # ---------- 收尾 ----------
    if major:
        scenes.append({
            "_id": next(scn_counter),
            "scene_name": major,
            "sub_scene_name": sub,
            "content": "\n".join(buf),
            "conversation": convs
        })
    return scenes

# ---------- 6. 主入口 ----------
def main(pdf="WanderingEarth2ScreenPlay.pdf", out_json="wandering_earth2_split.json"):
    doc = fitz.open(pdf)
    scenes = split_script(doc)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"✅ 解析完成：{len(scenes)} 条 → {out_json}")

if __name__ == "__main__":
    main()
