import json
import logging
import re
from typing import Dict, Any, List, Set, Tuple, Optional

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class TimelineParser:
    """
    时间线抽取器（规则候选 -> 加入提示词让 LLM 甄别；LLM 失败再兜底）
    使用 prompt_id = parse_timeline_prompt

    期望返回格式（字符串形式的 JSON，外部会再调用 correct_json_format）：
    {
        "timelines": ["2027年", "2058年", "二零二七年", ...]
    }

    约定：
    - 规则前置会抽取两类候选：
        1) 阿拉伯数字年份（如 2027年 / 2058—2075年 两端） -> 统一规范为 'YYYY年'，并做 1800–2300 过滤
        2) 中文数字年份（如 二零二七年/二〇二七年） -> 原样保留（仅清理空白与“年初/年末”等后缀）
    - 这些候选不会被直接采用，而是附加到提示词中，让模型“甄别、取舍或修正”
    - 如果 LLM 返回失败/异常，则用规则候选兜底（仍做一次轻量规范化、去重、保序）
    """

    def __init__(self, prompt_loader=None, llm=None):
        self.prompt_loader = prompt_loader
        self.llm = llm

        # 强制字段约束
        self.required_fields = ["timelines"]
        self.field_validators: Dict[str, Any] = {}

        # 修复模板
        self.repair_template = general_repair_template

        # ===== 预编译正则 =====
        # 年份范围：2058—2075年 / 2058-2075 / 2058–2075
        self._re_range = re.compile(r'(?P<y1>(?:1|2)\d{3})\s*[—–-]\s*(?P<y2>(?:1|2)\d{3})(?:\s*年)?')
        # 显式“年”：2027年 / 2027年末 / 2027 年初...
        self._re_year_with_nian = re.compile(
            r'(?P<y>(?<!\d)(?:1|2)\d{3})\s*年(?!\d)'
        )
        # 无“年”但跟“的”：2027的北京
        self._re_year_de = re.compile(r'(?P<y>(?<!\d)(?:1|2)\d{3})(?=的)')
        # 中文四位年份：二零二七年 / 二〇二七年（必须带“年”）
        cn_digit = r'[〇零一二三四五六七八九]'
        self._re_cn_year = re.compile(rf'(?P<cn>{cn_digit}{{4}})\s*年(初|末|底|间|份|季)?')

        # 合理年份范围（避免把页码、编号当年份）
        self._min_year, self._max_year = 1800, 2300

    # ============== 公共入口 ==============
    def call(self, params: str, **kwargs) -> str:
        """
        调用时间线抽取，保证返回 correct_json_format 处理后的 JSON 字符串

        params JSON:
            - text: 待抽取文本
            - existing: 之前已抽取的时间点（可选，用于提示 LLM 对齐）
        """
        try:
            params_dict = json.loads(params)
            text = params_dict.get("text", "") or ""
            existing = params_dict.get("existing", "")
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            error_result = {"error": f"参数解析失败: {str(e)}", "timelines": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

        try:
            # 1) 规则候选（仅作为 LLM 判断的输入，不直接采用）
            rule_candidates_list = self._rule_extract_timelines(text)

            # 2) 渲染主提示
            prompt_text = self.prompt_loader.render_prompt(
                prompt_id="parse_timeline_prompt",
                variables={"text": text}
            )

            # 3) 组装消息：把 existing 与 规则候选 都放入上下文中
            messages: List[Dict[str, str]] = []

            if existing:
                messages.append({
                    "role": "user",
                    "content": (
                        "这些是之前已经抽取的时间点信息，可供对齐参考（不一定完整/准确）：\n"
                        f"{existing}"
                    )
                })

            if rule_candidates_list:
                messages.append({
                    "role": "user",
                    "content": (
                        "下面是基于规则产生的候选年份，不一定正确，请甄别、取舍或修正：\n"
                        + json.dumps(rule_candidates_list, ensure_ascii=False)
                    )
                })

            messages.append({"role": "user", "content": prompt_text})

            # 4) 调用 LLM 并做格式保证
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=2,
                repair_template=self.repair_template,
                enable_thinking=False,
            )

            # 5) 成功：标准化 & 去重（保序）
            if status == "success":
                try:
                    data = json.loads(corrected_json)
                    model_out = data.get("timelines", [])
                except Exception:
                    model_out = []
                normalized = self._normalize_timelines(model_out)
                return correct_json_format(json.dumps({"timelines": normalized}, ensure_ascii=False))

            # 6) 失败：用规则候选兜底（仍规范化、保序）
            fallback = self._normalize_timelines(rule_candidates_list)
            return correct_json_format(json.dumps({"timelines": fallback}, ensure_ascii=False))

        except Exception as e:
            logger.error(f"时间线抽取过程中出现异常: {e}")
            error_result = {"error": f"时间线抽取失败: {str(e)}", "timelines": []}
            return correct_json_format(json.dumps(error_result, ensure_ascii=False))

    # ============== 规则抽取 ==============
    def _rule_extract_timelines(self, text: str) -> List[str]:
        """
        从原文中用规则抽取候选年份（去重保序的列表）：
        - 阿拉伯数字年份：规范为 'YYYY年'，做范围过滤
        - 中文数字年份（四个中文数字+年）：原样保留（仅清理空白与“年初/年末/年间/份/季”等后缀）
        - 年份区间只取两端（2058—2075年 -> 2058年, 2075年）
        """
        ordered: List[str] = []
        seen: Set[str] = set()

        # a) 区间（阿拉伯数字）：2058—2075(年) -> 取两端
        for m in self._re_range.finditer(text):
            y1, y2 = int(m.group('y1')), int(m.group('y2'))
            for y in (y1, y2):
                if self._is_valid_year(y):
                    norm = f"{y}年"
                    if norm not in seen:
                        seen.add(norm)
                        ordered.append(norm)

        # b) 显式“年”的年份（阿拉伯数字）：2027年 / 2027年末 / 2027 年初...
        for m in self._re_year_with_nian.finditer(text):
            y = int(m.group('y'))
            if self._is_valid_year(y):
                norm = f"{y}年"
                if norm not in seen:
                    seen.add(norm)
                    ordered.append(norm)

        # c) “YYYY的北京”（阿拉伯数字，缺少“年”，但紧跟“的”）
        for m in self._re_year_de.finditer(text):
            y = int(m.group('y'))
            if self._is_valid_year(y):
                norm = f"{y}年"
                if norm not in seen:
                    seen.add(norm)
                    ordered.append(norm)

        # d) 中文数字年份：二零二七年 / 二〇二七年 —— 原样保留（不转阿拉伯）
        for m in self._re_cn_year.finditer(text):
            raw = m.group(0)  # 包含“年”和可能的后缀
            # 去空白，如 “二 零 二 七 年”
            raw_norm = re.sub(r'\s+', '', raw)
            # 截到“年”
            raw_norm = re.sub(r'(年)(初|末|底|间|份|季)?$', r'\1', raw_norm)
            if raw_norm not in seen:
                seen.add(raw_norm)
                ordered.append(raw_norm)

        return ordered

    # ============== 规范化（用于模型输出或兜底） ==============
    def _normalize_timelines(self, items: List[Any]) -> List[str]:
        """
        标准化 timelines：
        - 中文四位数字+年：保持原样（仅去空白与截到“年”）
        - 阿拉伯数字形式：提取四位年并规范成 'YYYY年'（1800–2300 过滤）
        - 去重并保留首次出现顺序
        """
        ordered: List[str] = []
        seen: Set[str] = set()

        for it in items or []:
            s = str(it).strip()
            if not s:
                continue

            # 先匹配中文四位数字+年（保留原样）
            cm = re.search(r'([〇零一二三四五六七八九]{4})\s*年(初|末|底|间|份|季)?', s)
            if cm:
                raw_norm = re.sub(r'\s+', '', cm.group(0))
                raw_norm = re.sub(r'(年)(初|末|底|间|份|季)?$', r'\1', raw_norm)
                if raw_norm not in seen:
                    seen.add(raw_norm)
                    ordered.append(raw_norm)
                continue

            # 再看阿拉伯数字年份（四位）
            m = re.search(r'((?:1|2)\d{3})', s)
            if m:
                y = int(m.group(1))
                if self._is_valid_year(y):
                    norm = f"{y}年"
                    if norm not in seen:
                        seen.add(norm)
                        ordered.append(norm)

        return ordered

    # ============== 工具函数 ==============
    def _is_valid_year(self, y: int) -> bool:
        return self._min_year <= y <= self._max_year
