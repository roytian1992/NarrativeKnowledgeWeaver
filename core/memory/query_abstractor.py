from __future__ import annotations

import re
import hashlib
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from core.utils.prompt_loader import YAMLPromptLoader


class QueryAbstractor:
    """
    Convert user queries into task-level abstract forms so memory patterns are
    not tied to specific stories, characters, or document ids.
    """

    _RE_WS = re.compile(r"\s+")
    _RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _RE_DOC_ID = re.compile(r"\b(?:scene|chapter|doc|document)[_-]?\d+(?:[_-][A-Za-z0-9]+)*\b", re.IGNORECASE)
    _RE_CHUNK_ID = re.compile(r"\b\d+_seg_\d+_chunk_\d+\b", re.IGNORECASE)
    _RE_ENTITY_ID = re.compile(r"\b(?:ent|rel|int|pat)_[0-9a-f]{6,}\b", re.IGNORECASE)
    _RE_TIME = re.compile(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d{1,2}(?:am|pm)\b",
        re.IGNORECASE,
    )
    _RE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
    _RE_QUOTED = re.compile(r"[\"'“”‘’「」『』《》][^\"'“”‘’「」『』《》]{1,80}[\"'“”‘’「」『』《》]")
    _RE_TITLE_CASE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
    _ZH_SURNAMES = set(
        "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
        "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐罗薛伍"
        "余米贝姚孟顾尹江钟谭陆汪范金石廖贺倪汤滕殷毕郝邬安常乐于时傅皮卞齐"
        "康伍余元卜顾孟平黄和穆萧尉"
    )
    _RE_ZH_PERSON_NAME = re.compile(
        r"(?:(?<=^)|(?<=[\s，。！？、:：;；,.!?()（）和与跟对向在给]))"
        r"([赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
        r"戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐罗薛伍"
        r"余米贝姚孟顾尹江钟谭陆汪范金石廖贺倪汤滕殷毕郝邬安常乐于时傅皮卞齐"
        r"康伍余元卜顾孟平黄和穆萧尉][\u4e00-\u9fff]{1,2})"
        r"(?=(?:[\s，。！？、:：;；,.!?()（）和与跟对向在给问说道])|$)"
    )
    _RE_ZH_LOCATION = re.compile(
        r"[\u4e00-\u9fff]{2,10}(?:市|省|县|区|州|国|城|镇|村|总部|大楼|广场|走廊|会议室|会场)"
    )

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        prompt_loader: Optional[YAMLPromptLoader] = None,
        prompt_id: str = "memory/abstract_query_for_routing",
        abstraction_mode: str = "rule",
        max_cache_size: int = 2048,
    ) -> None:
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = str(prompt_id or "memory/abstract_query_for_routing").strip()
        self.abstraction_mode = str(abstraction_mode or "rule").strip().lower()
        if self.abstraction_mode not in {"rule", "llm", "hybrid"}:
            self.abstraction_mode = "rule"
        self.max_cache_size = max(128, int(max_cache_size))
        self._cache: dict[str, str] = {}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize(text: str) -> str:
        s = str(text or "").strip()
        s = QueryAbstractor._RE_WS.sub(" ", s)
        return s

    def _rule_abstract(self, query: str) -> str:
        s = self._normalize(query)
        if not s:
            return ""

        # Strong anonymization of identifiers and literals
        s = self._RE_URL.sub("[URL]", s)
        s = self._RE_EMAIL.sub("[EMAIL]", s)
        s = self._RE_DOC_ID.sub("[DOCUMENT]", s)
        s = self._RE_CHUNK_ID.sub("[CHUNK]", s)
        s = self._RE_ENTITY_ID.sub("[ID]", s)
        s = self._RE_TIME.sub("[TIME]", s)
        s = self._RE_QUOTED.sub("[MENTION]", s)
        s = self._RE_NUM.sub("[NUM]", s)

        # Replace likely proper-name spans in English.
        s = self._RE_TITLE_CASE.sub("[ENTITY]", s)

        # Replace likely Chinese person names by surname heuristic.
        s = self._RE_ZH_PERSON_NAME.sub("[CHARACTER]", s)
        s = self._RE_ZH_LOCATION.sub("[LOCATION]", s)

        s = self._normalize(s)
        return s

    def _llm_abstract(self, query: str, seed_abstract: str) -> str:
        if self.llm is None:
            return seed_abstract
        if self.prompt_loader is None:
            return seed_abstract
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                task_values={
                    "original_query": query,
                    "rule_abstract": seed_abstract,
                },
                static_values={},
                strict=True,
            )
        except Exception:
            return seed_abstract
        try:
            out = self.llm.invoke([
                SystemMessage(content="You are a strict abstraction rewriter for routing memory."),
                HumanMessage(content=prompt),
            ])
            txt = getattr(out, "content", str(out))
            if isinstance(txt, list):
                txt = " ".join(str(x) for x in txt)
            cand = self._normalize(str(txt or ""))
            if not cand:
                return seed_abstract
            # Apply one more anonymization pass as safety.
            return self._rule_abstract(cand)
        except Exception:
            return seed_abstract

    def abstract(self, query: str) -> str:
        q = self._normalize(query)
        if not q:
            return ""
        qh = self._hash(q)
        hit = self._cache.get(qh)
        if hit:
            return hit

        base = self._rule_abstract(q)
        if self.abstraction_mode in {"llm", "hybrid"}:
            out = self._llm_abstract(q, base)
        else:
            out = base

        if len(self._cache) >= self.max_cache_size:
            # cheap FIFO-like eviction
            try:
                self._cache.pop(next(iter(self._cache)))
            except Exception:
                self._cache.clear()
        self._cache[qh] = out
        return out
