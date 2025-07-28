from transformers import AutoModelForCausalLM, AutoTokenizer
import json
model_name = "/home/RoyTian/roytian/Qwen3/Qwen3-14B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0"
)

template = """请解析以下场景名称，提取其中的元数据信息：

场景名称：{scene_name}

**场景名称格式说明：**
- INT 表示内景（室内场景）
- EXT 表示外景（室外场景）
- 常见格式："场景编号、场景类型.地点.时间"
- 示例："1、EXT.学校操场.日" 或 "7、INT.客厅.夜" 或 "8、EXT/INT.日/夜.一组蒙太奇 8-4、山火漫野，一只失去家园的袋鼠茫然地站在山火前。"

**解析要求：**
- 识别场景类型：INT（内景）或 EXT（外景）
- 提取时间信息：日、夜、黄昏、清晨等
- 提取地点信息：主要位置和具体子位置，以及处于的宏观环境
- 判断是否为特殊场景：蒙太奇、回忆、梦境等

请按以下严格格式返回 JSON，不要添加换行和空格：
```json
{{
  "scene_type": "INT 或 EXT",
  "time_of_day": "时间信息（日/夜/黄昏/清晨等）",
  "environment": "非具体宏观环境（描述事件发生的物理或自然大背景，例如太空/沙漠等），非具体地点",
  "location": "主要具体地点，印度巴黎等国家、城市也算作此类",
  "sub_location": "具体子位置，例如实验室、食堂、观测舱等",
  "is_special_scene": "是否特殊场景（蒙太奇/回忆/梦境等），返还 true 或 false",
}}
...
**注意：**
- 仅输出符合 JSON 格式的内容，禁止添加额外解释、注释、自然语言说明。
- 如果某个字段无法确定，使用空字符串。
- 如果是特殊场景，地点（location/sub_location）和宏观环境（environment）设为空字符串。
- 对于包含子场景的场景名称，如："248、INT/EXT.日/夜.一组蒙太奇 248-2、INT.夜.巴黎 UEG 飞控中心", 如果包含非特殊场景的信息，即使出现"蒙太奇"等字段，也不算做特殊场景。
"""


prompt_template = {
  "id": "parse_scene_name_tool_prompt",
  "category": "tool",
  "name": "场景名称解析工具提示",
  "description": "用于提示LLM解析场景名称，提取元数据",
  "template": template,
  "variables": [
    {
      "name": "scene_name",
      "description": "待解析的场景名称"
    }
  ]
}

def get_meta_data(scene_name):
    messages = [
        {"role": "user", "content": prompt_template["template"].format(scene_name=scene_name)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content

    import json

import re

def correct_json_format(text: str) -> str:
    """
    把任意字符串清洗成纯 JSON 字符串（适合 json.loads 解析）。
    
    处理规则：
        1. 若存在 ```json ... ``` 代码块，提取首个代码块内容
        2. 若只找到 ```json 开围栏但没找到结尾，去掉开围栏并剥尾部 ```
        3. 若文本以 ``` 开头（非 json），剥去开关围栏
        4. 其余情况直接返回 strip 后的全文
        5. 把 'true/false' 替换成 Python 的 True/False 以便后续 eval 等
    """
    # ① 优先找完整代码块
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        body = m.group(1).strip()
    else:
        stripped = text.lstrip()

        # ② 只有开围栏（```json）无闭围栏
        if stripped.startswith("```json"):
            body = stripped[len("```json"):].lstrip()
            if body.endswith("```"):            # 万一尾部有多余 ```
                body = body[:-3].rstrip()
        else:
            # ③ 普通 ``` 围栏
            if stripped.startswith("```"):
                stripped = stripped[len("```"):].lstrip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
            body = stripped.strip()

    # ④ 替换 JSON 布尔值 → Python
    return body.replace("False", "false").replace("True", "true")


def get_scene_number(scene_name, sub_scene_name):
    pattern = r'(?:([\d\-]+)、)?'
    match_scene = re.match(pattern, scene_name)
    scene_number = match_scene.group(1)
    if len(sub_scene_name) > 0:
        match_sub_scene = re.match(pattern, sub_scene_name)
        sub_scene_number = match_sub_scene.group(1).split('-')[1]
    else:
        sub_scene_number = "1"
    return scene_number, sub_scene_number

    
json_file_path = "examples/wandering_earth2_split.json"
with open(json_file_path, 'r', encoding='utf-8') as f:
     data = json.load(f)


from tqdm import tqdm
import json

MAX_RETRIES = 3         # 允许的最大尝试次数
new_data     = []        # 成功结果
failed_logs  = []        # 保存仍然失败的条目信息，便于后续排查


for idx, item in enumerate(tqdm(data, desc="Processing scenes")):
    for attempt in range(1, MAX_RETRIES + 1):      # attempt: 1 或 2
        try:
            # ---------- 核心处理逻辑 ----------
            scene_name       = item.get("scene_name", "")
            sub_scene_name   = item.get("sub_scene_name", "")

            scene_number, sub_scene_number = get_scene_number(scene_name, sub_scene_name)
            if sub_scene_name:
                scene_name = sub_scene_name
            meta_raw = get_meta_data(scene_name)
            meta_data = json.loads(correct_json_format(meta_raw))

            if meta_data["is_special_scene"] == True:
                # 特殊场景：清空地点和宏观环境
                meta_data["location"] = ""
                meta_data["sub_location"] = ""
                meta_data["environment"] = ""

            # 合并并保留原字段
            new_item = {
                "scene_number":      scene_number,
                "sub_scene_number":  sub_scene_number,
                "meta_data":         meta_data,
                **item               # 保留原有键值
            }
            new_data.append(new_item)
            break   # 成功——退出重试循环

        except Exception as e:
            if attempt < MAX_RETRIES:
                # 第一次失败，打印并立即重试
                print(f"[Retry {attempt}] Error processing item {idx}: {e}")
            else:
                # 第二次仍失败，记录并放弃
                print(f"[Failed]  Item {idx} still failing after {MAX_RETRIES} attempts: {e}")
                failed_logs.append({"index": idx, "item": item, "error": str(e)})


with open("wandering_earth2_clean.json", 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)