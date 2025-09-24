# core/memory/vector_memory.py

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

from core.memory.base_memory import BaseMemory
from core.utils.config import KAGConfig
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    """
    Vector-based memory that stores and retrieves memories using a vector database
    with semantic similarity search. Falls back to a local JSON list if the vector
    database cannot be initialized or becomes unavailable.
    """

    def __init__(self, config: KAGConfig, category: Optional[str] = None) -> None:
        """
        Initialize the vector memory.

        Args:
            config: Global configuration object containing memory and embedding settings.
            category: Optional memory category used to namespace/prefix the persistence path.
        """
        super().__init__(config)
        self.config = config
        self.category = category or "default"
        self.memory_path = os.path.join(self.config.memory.memory_path, self.category)

        os.makedirs(self.memory_path, exist_ok=True)

        self.vector_db: Optional[Chroma] = None
        self.embedding_model: Any = None
        self.fallback_memory: List[Dict[str, Any]] = []
        self.use_fallback: bool = False

        self._init_vector_db()

    def _init_vector_db(self) -> None:
        """
        Initialize the vector database and embedding model.

        If initialization fails, enable the JSON fallback mode.
        """
        try:
            if self.config.graph_embedding.provider != "local":
                # Uses a project-specific OpenAI embedding wrapper that implements the LangChain interface.
                from core.model_providers.openai_embedding import OpenAIEmbeddingModel

                self.embedding_model = OpenAIEmbeddingModel(self.config.graph_embedding)
                self.vector_db = Chroma(
                    collection_name="memory",
                    embedding_function=self.embedding_model,
                    persist_directory=self.memory_path,
                )
            else:
                # Prefer the LangChain HuggingFaceEmbeddings wrapper for compatibility with Chroma.
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings

                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name=self.config.graph_embedding.model_name
                    )
                except Exception:
                    # Fallback to raw SentenceTransformer if wrapper is unavailable.
                    from sentence_transformers import SentenceTransformer

                    st_model = SentenceTransformer(self.config.graph_embedding.model_name)

                    # Minimal adapter to match LangChain Embeddings interface.
                    class _STAdapter:
                        def __init__(self, model):
                            self.model = model

                        def embed_documents(self, texts: List[str]) -> List[List[float]]:
                            return self.model.encode(texts, normalize_embeddings=True).tolist()

                        def embed_query(self, text: str) -> List[float]:
                            return self.model.encode([text], normalize_embeddings=True)[0].tolist()

                    self.embedding_model = _STAdapter(st_model)

                self.vector_db = Chroma(
                    collection_name="memory",
                    embedding_function=self.embedding_model,
                    persist_directory=self.memory_path,
                )

            logger.debug("VectorMemory initialized successfully at %s", self.memory_path)
            self.use_fallback = False

        except Exception as e:
            logger.exception("Failed to initialize vector database: %s", str(e))
            self.fallback_memory = []
            self.use_fallback = True

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a memory item (explicit text plus optional metadata).

        Args:
            text: The text to be embedded for semantic retrieval.
            metadata: Optional structured metadata for filtering or display.
        """
        metadata = dict(metadata or {})

        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()

        if self.use_fallback:
            metadata["text"] = text  # preserve the raw text in fallback mode
            self.fallback_memory.append(metadata)
            self._save_fallback()
            logger.debug("Added memory to JSON fallback (count=%d).", len(self.fallback_memory))
            return

        try:
            assert self.vector_db is not None
            self.vector_db.add_texts(texts=[text], metadatas=[metadata])
            self.vector_db.persist()
            logger.debug("Added memory to vector DB and persisted.")
        except Exception as e:
            logger.exception("Failed to add memory to vector DB: %s", str(e))
            # Switch to fallback and re-add the same record
            self.use_fallback = True
            self.fallback_memory = []
            self.add(text, metadata)

    def get(self, query: Optional[str] = None, k: int = 5) -> List[Any]:
        """
        Retrieve memory items.

        Args:
            query: Optional text query for semantic search. If omitted, returns recent items.
            k: Maximum number of items to return.

        Returns:
            A list of results. In vector mode, returns LangChain Document objects.
            In fallback mode, returns a list of metadata dicts (each containing a "text" field).
        """
        if self.use_fallback:
            logger.debug("Using JSON fallback; returning last %d memories.", k)
            return self.fallback_memory[-k:]

        try:
            assert self.vector_db is not None
            if query:
                results = self.vector_db.similarity_search(query, k=k)
            else:
                # Heuristic "recent" retrieval by searching with a neutral query.
                # (Note: If you need strict recency, store timestamps as metadata and sort externally.)
                results = self.vector_db.similarity_search("recent memories", k=k)
            logger.debug("Retrieved %d memory items from vector DB.", len(results))
            return results
        except Exception as e:
            logger.exception("Failed to retrieve memories from vector DB: %s", str(e))
            return []

    def clear(self) -> None:
        """
        Clear all memories.

        In fallback mode, clears the JSON list. In vector mode, attempts to delete all
        items from the Chroma collection. Falls back to reinitialization if needed.
        """
        if self.use_fallback:
            self.fallback_memory = []
            self._save_fallback()
            logger.info("Cleared JSON fallback memories.")
            return

        try:
            assert self.vector_db is not None

            # Different Chroma/LangChain versions expose different deletion methods.
            # Try a full-collection delete first, then fall back to a blanket where-filter.
            if hasattr(self.vector_db, "delete_collection"):
                self.vector_db.delete_collection()  # type: ignore[attr-defined]
                logger.info("Deleted vector DB collection.")
            else:
                try:
                    # Delete all documents with an empty filter.
                    self.vector_db.delete(where={})  # type: ignore[call-arg]
                    logger.info("Deleted all documents from vector DB collection via where={}.")
                except Exception:
                    # As a last resort, recreate the store directory.
                    logger.warning("Falling back to re-initialization after deletion failure.")
            # Reinitialize to ensure a clean state.
            self._init_vector_db()

        except Exception as e:
            logger.exception("Failed to clear vector DB: %s", str(e))

    def save(self) -> None:
        """
        Persist memories to disk.
        """
        if self.use_fallback:
            self._save_fallback()
            logger.debug("Persisted JSON fallback memories.")
            return

        try:
            assert self.vector_db is not None
            self.vector_db.persist()
            logger.debug("Persisted vector DB to disk.")
        except Exception as e:
            logger.exception("Failed to persist vector DB: %s", str(e))

    def load(self) -> None:
        """
        Load memories from disk.

        In vector mode, Chroma is loaded during initialization. In fallback mode,
        load the JSON file if present.
        """
        if self.use_fallback:
            self._load_fallback()
            logger.debug("Loaded JSON fallback memories.")
        # Vector DB is already loaded on initialization.

    def _save_fallback(self) -> None:
        """
        Save fallback memories to a JSON file.
        """
        fallback_path = os.path.join(self.memory_path, "fallback_memory.json")
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(self.fallback_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("Failed to save fallback JSON: %s", str(e))

    def _load_fallback(self) -> None:
        """
        Load fallback memories from a JSON file.
        """
        fallback_path = os.path.join(self.memory_path, "fallback_memory.json")
        if os.path.exists(fallback_path):
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    self.fallback_memory = json.load(f)
            except Exception as e:
                logger.exception("Failed to load fallback JSON: %s", str(e))
                self.fallback_memory = []
        else:
            self.fallback_memory = []
