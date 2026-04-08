"""
core/memory/base_memory.py

Abstract base class for all memory implementations.
Concrete subclasses:
  - ExtractionMemoryStore  (core/memory/extraction_store.py)
  - (future) ToolCallMemoryStore
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemoryStore(ABC):
    """
    Minimal ABC for a persistent memory store.

    Every subclass must implement:
      - add(entry)   → str        insert/merge an entry, return its id
      - query(text)  → str        retrieve relevant entries as a formatted hint string
      - flush()                   persist in-memory state to disk
    """

    @abstractmethod
    def add(self, entry: Dict[str, Any]) -> str:
        """Add (or merge) a memory entry. Returns the assigned id."""

    @abstractmethod
    def query(self, text: str, **kwargs) -> str:
        """
        Retrieve relevant memory entries for the given text.
        Returns a formatted string suitable for injecting into a prompt.
        Returns empty string when no relevant entries exist.
        """

    @abstractmethod
    def flush(self) -> None:
        """Persist any in-memory dirty state to the backing store."""

    # Optional convenience helpers with default no-op implementations
    def delete(self, entry_id: str) -> None:
        pass

    def update(self, entry_id: str, **fields: Any) -> None:
        pass

    def mark_doc_processed(self) -> None:
        pass

    def load_from_disk(self) -> None:
        pass
