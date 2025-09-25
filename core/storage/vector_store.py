"""
Vector database storage module

ChromaDB-based vector storage and semantic search.
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
import os
import logging
import chromadb
from chromadb.config import Settings
from core.models.data import Document
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database storage based on ChromaDB."""

    def __init__(self, config: KAGConfig, category: str = "documents") -> None:
        """
        Initialize the VectorStore.

        Args:
            config: Global configuration object.
            category: Collection name/category. Also used as the subdirectory for persistence.
        """
        self.config = config
        self.vector_store_path = os.path.join(self.config.storage.vector_store_path, category)
        self.vector_store_name = category  # collection name = category
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client, collection, and embedding model."""
        try:
            # Initialize ChromaDB persistent client
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection (named by category)
            self.collection = self.client.get_or_create_collection(
                name=self.vector_store_name,
                metadata={"description": f"{self.vector_store_name} vectordb"},
            )

            # Initialize embedding model
            if self.config.vectordb_embedding.provider != "local":
                # Project-specific OpenAI embedding wrapper; expected to expose `.encode(texts)`
                from core.model_providers.openai_embedding import OpenAIEmbeddingModel

                self.embedding_model = OpenAIEmbeddingModel(self.config.vectordb_embedding)
            else:
                # SentenceTransformer exposes `.encode(texts)`
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer(self.config.vectordb_embedding.model_name)

            logger.info(
                "Vector DB initialized: path=%s, collection=%s",
                self.vector_store_path,
                self.vector_store_name,
            )

        except Exception as e:
            logger.exception("Failed to initialize vector DB: %s", str(e))
            self.client = None
            self.collection = None

    def _ensure_collection(self) -> None:
        """Ensure the collection is available; if deleted, recreate it."""
        if not self.client:
            return
        try:
            _ = self.collection.count()
        except Exception:
            logger.warning("Collection handle invalid; re-creating collection '%s'.", self.vector_store_name)
            self.collection = self.client.get_or_create_collection(
                name=self.vector_store_name,
                metadata={"description": f"{self.vector_store_name} vectordb"},
            )

    @staticmethod
    def _iter_batches(items: List[Any], batch_size: int) -> Iterable[Tuple[int, List[Any]]]:
        """Yield (start_index, batch_list) for a list."""
        for i in range(0, len(items), batch_size):
            yield i, items[i : i + batch_size]

    def store_documents(self, documents: List[Document], batch_size: int = 500) -> None:
        """
        Store documents into the vector database.
        If the number of documents exceeds `batch_size`, they are written in batches.

        Args:
            documents: List of `Document` objects to upsert.
            batch_size: Number of documents per batch (default: 500).
        """
        if not self.client or not self.collection:
            logger.warning("Vector DB is not initialized; skipping vector storage.")
            return

        if not documents:
            logger.info("No documents to store.")
            return

        if batch_size <= 0:
            batch_size = 500  # fallback

        self._ensure_collection()

        total = len(documents)
        success = 0
        failed_batches: List[Tuple[int, int, str]] = []

        for start_idx, batch in self._iter_batches(documents, batch_size):
            try:
                ids: List[str] = []
                texts: List[str] = []
                metadatas: List[Dict[str, Any]] = []

                for doc in batch:
                    ids.append(str(doc.id))
                    texts.append(doc.content)

                    # Chroma metadata supports scalar values; convert others to string.
                    md: Dict[str, Any] = {}
                    for key, value in (doc.metadata or {}).items():
                        if isinstance(value, (str, int, float, bool)):
                            md[key] = value
                        else:
                            md[key] = str(value)
                    metadatas.append(md)

                # Compute embeddings
                embeddings = self.embedding_model.encode(texts)
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()

                # Upsert batch
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                success += len(batch)
                end_idx = min(start_idx + len(batch), total)
                logger.info("Batch upsert succeeded: %d-%d (%d items)", start_idx, end_idx - 1, len(batch))

            except Exception as e:
                end_idx = min(start_idx + len(batch), total)
                failed_batches.append((start_idx, end_idx, str(e)))
                logger.exception("Batch upsert failed: %d-%d. Error: %s", start_idx, end_idx - 1, e)

        # Summary
        if failed_batches:
            logger.warning("Summary: %d/%d items succeeded, %d failed batches.", success, total, len(failed_batches))
            for s, e, msg in failed_batches:
                logger.warning("  - Batch %d-%d failed: %s", s, e - 1, msg)
        else:
            logger.info("All upserts succeeded: %d items (batch size %d).", success, batch_size)

    def search(self, query: str, limit: int = 5) -> List[Document]:
        """
        Semantic search by query text.

        Args:
            query: Search query text.
            limit: Max number of results to return.

        Returns:
            A list of `Document` objects with `metadata["similarity_score"]` when available.
        """
        if not self.client or not self.collection:
            return []

        self._ensure_collection()

        try:
            q = self.embedding_model.encode([query])
            if hasattr(q, "tolist"):
                q = q.tolist()
            query_embedding = q[0] if isinstance(q, list) and len(q) > 0 else q

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],  # ids are returned by default
            )

            documents: List[Document] = []
            if not results or not results.get("ids"):
                return documents

            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results.get("distances", [[None] * len(ids)])[0]

            for i in range(len(ids)):
                meta = metas[i] or {}
                if dists[i] is not None:
                    # Convert distance to a similarity score in [0, 1] heuristically
                    meta["similarity_score"] = 1.0 - float(dists[i])
                documents.append(Document(id=ids[i], content=docs[i], metadata=meta))

            return documents

        except Exception as e:
            logger.exception("Semantic search failed: %s", str(e))
            return []

    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Document]:
        """
        Filter documents by metadata using Chroma's `where` clause.

        Args:
            metadata_filter: A dict of metadata conditions. Non-dict values are wrapped as {"$eq": value}.
            limit: Max number of results to return.

        Returns:
            A list of `Document` objects.
        """
        if not self.client or not self.collection:
            return []

        self._ensure_collection()

        try:
            where_conditions: Dict[str, Any] = {}
            for key, value in (metadata_filter or {}).items():
                if isinstance(value, dict):
                    where_conditions[key] = value
                else:
                    where_conditions[key] = {"$eq": value}

            results = self.collection.get(
                where=where_conditions,
                limit=limit,
                include=["documents", "metadatas"],  # ids are returned by default
            )

            documents: List[Document] = []
            if not results or not results.get("ids"):
                return documents

            for i, doc_id in enumerate(results["ids"]):
                documents.append(
                    Document(
                        id=doc_id,
                        content=results["documents"][i],
                        metadata=results["metadatas"][i] or {},
                    )
                )
            return documents

        except Exception as e:
            logger.exception("Metadata search failed: %s", str(e))
            return []

    def search_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """
        Retrieve documents by a list of IDs.

        Args:
            doc_ids: List of document IDs.

        Returns:
            A list of `Document` objects.
        """
        if not self.client or not self.collection:
            logger.warning("Vector DB is not initialized; cannot query by IDs.")
            return []
        if not doc_ids:
            return []

        self._ensure_collection()

        try:
            result = self.collection.get(
                ids=[str(x) for x in doc_ids],
                include=["documents", "metadatas"],  # ids are returned by default
            )

            documents: List[Document] = []
            if not result or not result.get("ids"):
                return documents

            for i, doc_id in enumerate(result["ids"]):
                documents.append(
                    Document(
                        id=doc_id,
                        content=result["documents"][i],
                        metadata=result["metadatas"][i] or {},
                    )
                )
            return documents

        except Exception as e:
            logger.exception("Batch ID lookup failed: %s", str(e))
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics and configuration snapshot.

        Returns:
            A dict containing connection status, path, collection name, document count,
            and embedding configuration (mirrors original field names for compatibility).
        """
        if not self.client or not self.collection:
            return {"status": "disconnected"}

        self._ensure_collection()

        try:
            count = self.collection.count()
            # Keep the original keys as in the provided code to ensure I/O compatibility.
            return {
                "status": "connected",
                "path": self.vector_store_path,
                "collection": self.vector_store_name,
                "document_count": count,
                "embedding_model": self.config.embedding.model_name,
                "embedding_provider": self.config.embedding.provider,
                "embedding_dim": self.config.embedding.dimensions,
            }
        except Exception as e:
            logger.exception("Failed to fetch stats: %s", str(e))
            return {"status": "error", "error": str(e)}

    def delete_collection(self) -> None:
        """
        Delete the current collection and recreate an empty one with the same name.
        Keeps the instance usable after deletion.
        """
        if self.client and self.collection:
            try:
                self.client.delete_collection(self.vector_store_name)
                logger.info("Vector collection deleted: %s", self.vector_store_name)
                # Recreate the collection to keep this instance usable
                self.collection = self.client.get_or_create_collection(
                    name=self.vector_store_name,
                    metadata={"description": f"{self.vector_store_name} vectordb"},
                )
            except Exception as e:
                logger.exception("Failed to delete vector collection: %s", str(e))
