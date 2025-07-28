# kag/agent/base_agent.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

class ReflectionMemory(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[str]]: ...
    @abstractmethod
    def store(self, context: str, reflection: Dict[str, Any]) -> None: ...

class BaseAgent(ABC):
    def __init__(self,
                 reflection_memory: ReflectionMemory,
                 score_threshold: float = 7.0,
                 max_retry: int = 3):
        self.ref_mem         = reflection_memory
        self.score_threshold = score_threshold
        self.max_retry       = max_retry
        self.prev_extraction: Optional[Dict[str, Any]] = None

    # ────────────────── 抽象方法 ────────────────── #
    @abstractmethod
    def _extract(self,
                 text: str,
                 issues: Optional[str],
                 suggestions: Optional[str],
                 prev_extraction: Optional[Dict[str, Any]]
                 ) -> Dict[str, Any]:
        """实体 + 关系抽取"""
        raise NotImplementedError

    @abstractmethod
    def _reflect(self,
                 extraction: Dict[str, Any],
                 text: str
                 ) -> Dict[str, Any]:
        """对抽取结果打分并给改进建议"""
        raise NotImplementedError

    # ────────────────── 主入口 ────────────────── #
    def run(self, text: str) -> Dict[str, Any]:
        issues_hist, sugg_hist = self.ref_mem.retrieve(text, k=5)
        issues_str = "\n".join(issues_hist) or None
        sugg_str   = "\n".join(sugg_hist)  or None

        best, best_score, best_refl = None, -1, {}
        prev = self.prev_extraction

        for _ in range(self.max_retry):
            extraction = self._extract(text, issues_str, sugg_str, prev)
            reflection = self._reflect(extraction, text)
            score      = float(reflection.get("score", 10))

            if extraction["entities"] and extraction["relations"] and score > best_score:
                best, best_score, best_refl = extraction | {"score": score}, score, reflection

            if score >= self.score_threshold:
                break

            prev       = extraction
            issues_str = "\n".join(reflection.get("current_issues", [])) or None
            sugg_str   = "\n".join(reflection.get("suggestions", []))   or None

        self.ref_mem.store(text, best_refl)
        self.prev_extraction = best
        return {"extraction": best, "reflection": best_refl}
