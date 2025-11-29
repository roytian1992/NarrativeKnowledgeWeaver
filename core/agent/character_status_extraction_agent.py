# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, Dict, List, Optional, Union

from core.utils.config import KAGConfig
from core.utils.format import correct_json_format
from core.utils.prompt_loader import PromptLoader
from core.builder.manager.supplementary_manager import SupplementaryExtractor

logger = logging.getLogger(__name__)


def _character_results_to_text(results: List[Dict[str, Any]]) -> str:
    """
    将抽取的角色结果转换成反思用的纯文本描述。
    格式示例：
        角色A：站在门口沉默不语，表情紧张。
        角色B：坐在沙发上翻看文件，偶尔抬头询问。
    """
    lines: List[str] = []
    for item in results or []:
        name = (item.get("name") or "").strip()
        status = (item.get("status") or "").strip()
        if not name and not status:
            continue
        if name and status:
            lines.append(f"{name}：{status}")
        elif name:
            lines.append(f"{name}：")
        else:
            lines.append(status)
    return "\n".join(lines)


class CharacterStatusExtractionAgent:
    """
    基于 LangChain/LMM 调用的「角色状态抽取 + 反思」小 Agent。

    流程（单场景）：
      1) extract：根据 scene_contents / character_list 抽取角色状态；
      2) reflect：对抽取结果做内容质量反思，输出 feedbacks(list) + score(0-10)；
      3) 若 score >= min_score 或达到 max_retries，停止；
         否则将累计 feedbacks 以 `\"\\n\".join(feedbacks)` 的形式
         作为 feedbacks 变量重新喂给 extract，进入下一轮。

    抽取输出期望格式（由 CharacterStatusExtractor 提供）：
      {
        "results": [
          {"name": "...", "status": "..."},
          ...
        ]
      }

    反思输出期望格式（由 CharacterStatusReflector 提供）：
      {
        "feedbacks": ["...", "..."],
        "score": 0-10
      }
    """

    def __init__(
        self,
        config: KAGConfig,
        llm,
        prompt_loader: Optional[PromptLoader] = None,
        min_score: float = 7.0,
        max_retries: int = 3,
        enable_thinking: bool = True,
    ):
        self.config = config
        self.llm = llm
        self.enable_thinking = enable_thinking

        if prompt_loader is None:
            prompt_dir = self.config.knowledge_graph_builder.prompt_dir
            prompt_loader = PromptLoader(prompt_dir)

        self.supplementary_extractor = SupplementaryExtractor(
            config=self.config,
            llm=self.llm,
            prompt_loader=prompt_loader,
        )

        # 控制参数
        self.min_score = min_score
        self.max_retries = max_retries

    # ----------------- 内部工具 -----------------
    @staticmethod
    def _normalize_character_list(character_list: Union[str, List[str]]) -> str:
        """
        将 character_list 统一转成字符串，方便塞进 prompt。
        - 若已是 str，直接返回；
        - 若是 List[str]，用 ' / ' 连接；
        """
        if isinstance(character_list, str):
            return character_list
        if isinstance(character_list, list):
            return " / ".join([str(x) for x in character_list if x])
        return ""

    @staticmethod
    def _parse_extraction_output(raw: str) -> Dict[str, Any]:
        """
        将 CharacterStatusExtractor 的输出 JSON 字符串解析成 dict，
        并容错 correct_json_format。
        """
        try:
            data = json.loads(correct_json_format(raw))
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}

    @staticmethod
    def _parse_reflection_output(raw: str) -> Dict[str, Any]:
        """
        将 CharacterStatusReflector 的输出 JSON 字符串解析成 dict。
        期望结构：
          { "feedbacks": [...], "score": number }
        """
        try:
            data = json.loads(correct_json_format(raw))
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}

    # ----------------- 主流程：带反思的抽取 -----------------
    def run(
        self,
        scene_contents: str,
        character_list: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        同步入口：对单个场景做「抽取 + 反思循环」。

        Args:
            scene_contents: 场景完整文本。
            character_list: 参考角色列表，可以是 str 或 List[str]。

        Returns:
            {
              "results": [...],          # 最终选定的角色状态列表
              "score": float,            # 最终得分
              "attempts": int,           # 实际尝试轮数
              "feedbacks": [...],        # 累积的所有反馈
              "raw_extractions": [...],  # 每一轮原始抽取结果（dict）
              "raw_reflections": [...]   # 每一轮原始反思结果（dict）
            }
        """
        clist_str = self._normalize_character_list(character_list)

        all_feedbacks: List[str] = []
        raw_extractions: List[Dict[str, Any]] = []
        raw_reflections: List[Dict[str, Any]] = []

        best_results: List[Dict[str, Any]] = []
        best_score: float = -1.0

        attempts = 0

        while attempts < self.max_retries:
            attempts += 1

            # 1) 抽取：feedbacks 用 "\n".join(all_feedbacks)，第一轮为空字符串
            feedbacks_for_prompt = "\n".join(all_feedbacks) if all_feedbacks else ""
            extraction_raw = self.supplementary_extractor.extract_character_status(
                scene_contents=scene_contents,
                character_list=clist_str,
                feedbacks=feedbacks_for_prompt,
                enable_thinking=self.enable_thinking,
            )
            extraction_data = self._parse_extraction_output(extraction_raw)
            raw_extractions.append(extraction_data)

            # 当前轮的结果
            current_results = extraction_data.get("results", [])
            if not isinstance(current_results, list):
                current_results = []

            # 2) 将抽取结果转成纯文本，喂给反思器
            extracted_text = _character_results_to_text(current_results)

            reflection_raw = self.supplementary_extractor.reflect_character_status(
                extracted_results=extracted_text,
                character_list=clist_str,
                scene_contents=scene_contents,
                enable_thinking=self.enable_thinking,
            )
            reflection_data = self._parse_reflection_output(reflection_raw)
            raw_reflections.append(reflection_data)

            feedbacks = reflection_data.get("feedbacks", []) or []
            score = reflection_data.get("score", 0.0)

            # 合并反馈（用于下一轮）
            for fb in feedbacks:
                if isinstance(fb, str) and fb.strip():
                    all_feedbacks.append(fb.strip())

            # 记录最优结果
            try:
                score_val = float(score)
            except Exception:
                score_val = 0.0

            if score_val > best_score:
                best_score = score_val
                best_results = current_results

            logger.debug(
                f"[CharacterStatusExtractionAgent] attempt={attempts}, "
                f"score={score_val}, feedbacks_this_round={len(feedbacks)}"
            )

            # 达到阈值则提前停止
            if score_val >= self.min_score:
                break

        return {
            "results": best_results,
            "score": best_score if best_score >= 0 else 0.0,
            "attempts": attempts,
            "feedbacks": all_feedbacks,
            "raw_extractions": raw_extractions,
            "raw_reflections": raw_reflections,
        }
