# retriever/sparse_retriever.py

# -*- coding: utf-8 -*-
"""
BM25 Keyword Retriever (Simplified Wrapper)
===========================================

This module provides a thin wrapper around LangChain's
`BM25Retriever` (community implementation). It standardizes
the input/output format to use `langchain_core.documents.Document`
objects and hides internal details for stability.

Key features:
-------------
- Accepts and returns `Document` objects (with `page_content` and `metadata`).
- Supports optional Chinese preprocessing (custom tokenizer).
- Adds consistent metadata fields: `source="bm25"` and
  `similarity_score=None` (BM25 in LangChain does not expose scores).
- Keeps implementation lightweight, without a direct dependency on
  `rank_bm25.BM25Okapi`.

Class:
------
    KeywordBM25Retriever
        A simple BM25 keyword retriever with a standardized interface.

Usage example:
--------------
    >>> from langchain_core.documents import Document
    >>> docs = [Document(page_content="The quick brown fox", metadata={})]
    >>> retriever = KeywordBM25Retriever(docs, k_default=5)
    >>> results = retriever.retrieve("brown fox")
    >>> for r in results:
    ...     print(r.page_content, r.metadata)
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LCBM25Retriever


class KeywordBM25Retriever:
    """
    Simple wrapper around LangChain's BM25Retriever.

    Provides:
    - Unified Document input/output.
    - Optional Chinese preprocessing via `zh_preprocess`.
    - Default top-k retrieval control.
    """

    def __init__(
        self,
        documents: List[Document],
        *,
        zh_preprocess=None,
        k_default: int = 10,
    ):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of `Document` objects to index.
            zh_preprocess: Optional preprocessing function for Chinese text.
            k_default: Default number of documents to return.
        """
        self.k_default = k_default
        self._bm25 = LCBM25Retriever.from_documents(
            documents=documents,
            preprocess_func=zh_preprocess,
        )
        self._bm25.k = k_default

    def retrieve(self, query: str, *, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents relevant to a query using BM25.

        Args:
            query: Input text query.
            k: Number of documents to return (defaults to self.k_default).

        Returns:
            List of `Document` objects with updated metadata:
            - `"source": "bm25"`
            - `"similarity_score": None` (BM25 does not expose raw scores).
        """
        k = int(k or self.k_default)
        self._bm25.k = k
        candidates: List[Document] = self._bm25.get_relevant_documents(query) or []

        out: List[Document] = []
        for d in candidates[:k]:
            md = dict(d.metadata or {})
            md["source"] = "bm25"
            md["similarity_score"] = None
            out.append(Document(page_content=d.page_content, metadata=md))
        return out
