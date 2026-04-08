"""
Vector database storage module

ChromaDB-based vector storage and semantic search.
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
import os
import logging
import threading
import chromadb
from chromadb.config import Settings
from core.models.data import Document
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)
_VECTORSTORE_INIT_LOCK = threading.Lock()


class VectorStore:
    """Vector database storage based on ChromaDB."""

    def __init__(self, config: KAGConfig, category: str = "documents", load_embedding_model: bool = True) -> None:
        """
        Initialize the VectorStore.

        Args:
            config: Global configuration object.
            category: Collection name/category. Also used as the subdirectory for persistence.
            load_embedding_model: Whether to initialize the embedding model for text-to-vector encoding.
        """
        self.config = config
        self.vector_store_path = os.path.join(self.config.storage.vector_store_path, category)
        self.vector_store_name = category  # collection name = category
        self.load_embedding_model = bool(load_embedding_model)
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client, collection, and embedding model."""
        try:
            # Chroma 1.x may fail under concurrent PersistentClient initialization
            # against the same persistence directory, so serialize client setup.
            with _VECTORSTORE_INIT_LOCK:
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

            # Initialize embedding model only when this store needs to encode raw text.
            if self.load_embedding_model:
                emb_cfg = getattr(self.config, "embedding", None)
                if emb_cfg is None:
                    raise ValueError("config.embedding is required for VectorStore")

                if emb_cfg.provider != "local":
                    # Project-specific OpenAI embedding wrapper; expected to expose `.encode(texts)`
                    from core.model_providers.openai_embedding import OpenAIEmbeddingModel

                    self.embedding_model = OpenAIEmbeddingModel(emb_cfg)
                else:
                    # SentenceTransformer exposes `.encode(texts)`
                    from sentence_transformers import SentenceTransformer

                    self.embedding_model = SentenceTransformer(emb_cfg.model_name)

            logger.info(
                "Vector DB initialized: path=%s, collection=%s, embedding_model=%s",
                self.vector_store_path,
                self.vector_store_name,
                "loaded" if self.embedding_model is not None else "skipped",
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

    @staticmethod
    def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize metadata to Chroma-compatible scalar values."""
        md: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)):
                md[key] = value
            elif value is None:
                continue
            else:
                md[key] = str(value)
        return md

    @staticmethod
    def _normalize_embedding(embedding: Any) -> Optional[List[float]]:
        """Normalize a vector into a 1D float list."""
        if embedding is None:
            return None
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        if not isinstance(embedding, list):
            return None
        try:
            return [float(x) for x in embedding]
        except Exception:
            return None

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
        if self.embedding_model is None:
            logger.warning("Embedding model is not initialized for collection '%s'; cannot encode documents.", self.vector_store_name)
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
                    metadatas.append(self._normalize_metadata(doc.metadata or {}))

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

    def upsert_records(self, records: List[Dict[str, Any]], batch_size: int = 500) -> None:
        """
        Upsert records with externally supplied embeddings.

        Expected record format:
            {
                "id": str,
                "content": str,
                "metadata": {...},
                "embedding": List[float],
            }
        """
        if not self.client or not self.collection:
            logger.warning("Vector DB is not initialized; skipping record upsert.")
            return
        if not records:
            return
        if batch_size <= 0:
            batch_size = 500

        self._ensure_collection()

        normalized: List[Dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            record_id = str(row.get("id", "") or "").strip()
            if not record_id:
                continue
            embedding = self._normalize_embedding(row.get("embedding"))
            if not embedding:
                continue
            content = str(row.get("content", "") or row.get("text", "") or record_id)
            normalized.append(
                {
                    "id": record_id,
                    "content": content,
                    "metadata": self._normalize_metadata(row.get("metadata") or {}),
                    "embedding": embedding,
                }
            )
        if not normalized:
            return

        total = len(normalized)
        success = 0
        failed_batches: List[Tuple[int, int, str]] = []
        for start_idx, batch in self._iter_batches(normalized, batch_size):
            try:
                self.collection.upsert(
                    ids=[row["id"] for row in batch],
                    documents=[row["content"] for row in batch],
                    metadatas=[row["metadata"] for row in batch],
                    embeddings=[row["embedding"] for row in batch],
                )
                success += len(batch)
            except Exception as e:
                end_idx = min(start_idx + len(batch), total)
                failed_batches.append((start_idx, end_idx, str(e)))
                logger.exception("Vector record upsert failed: %d-%d. Error: %s", start_idx, end_idx - 1, e)

        if failed_batches:
            logger.warning("Record upsert summary: %d/%d succeeded, %d failed batches.", success, total, len(failed_batches))
        else:
            logger.info("Vector record upsert succeeded: %d items.", success)

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
        if self.embedding_model is None:
            logger.warning("Embedding model is not initialized for collection '%s'; cannot encode query text.", self.vector_store_name)
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

    def search_by_embedding(
        self,
        query_embedding: Any,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Semantic search by a precomputed query embedding.
        """
        if not self.client or not self.collection:
            return []

        self._ensure_collection()

        query_vector = self._normalize_embedding(query_embedding)
        if not query_vector:
            return []

        try:
            where_conditions: Optional[Dict[str, Any]] = None
            if metadata_filter:
                where_conditions = {}
                for key, value in metadata_filter.items():
                    if isinstance(value, dict):
                        where_conditions[key] = value
                    else:
                        where_conditions[key] = {"$eq": value}

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where_conditions,
                include=["documents", "metadatas", "distances"],
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
                    meta["similarity_score"] = 1.0 - float(dists[i])
                documents.append(Document(id=ids[i], content=docs[i], metadata=meta))
            return documents

        except Exception as e:
            logger.exception("Embedding search failed: %s", str(e))
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

    def get_records_by_ids(
        self,
        doc_ids: List[str],
        *,
        include_embeddings: bool = False,
        batch_size: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve raw vector-store records by IDs.
        """
        if not self.client or not self.collection:
            logger.warning("Vector DB is not initialized; cannot query records by IDs.")
            return []
        if not doc_ids:
            return []
        if batch_size <= 0:
            batch_size = 500

        self._ensure_collection()

        normalized_ids: List[str] = []
        seen = set()
        for doc_id in doc_ids:
            text = str(doc_id or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized_ids.append(text)
        if not normalized_ids:
            return []

        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        def _as_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if hasattr(value, "tolist"):
                converted = value.tolist()
                if isinstance(converted, list):
                    return converted
                return [converted]
            try:
                return list(value)
            except TypeError:
                return [value]

        rows_by_id: Dict[str, Dict[str, Any]] = {}
        for _start_idx, batch in self._iter_batches(normalized_ids, batch_size):
            try:
                result = self.collection.get(ids=batch, include=include_fields)
            except Exception as e:
                logger.exception("Raw ID lookup failed: %s", str(e))
                continue
            if not result or not result.get("ids"):
                continue

            ids = _as_list(result.get("ids"))
            docs = _as_list(result.get("documents"))
            metas = _as_list(result.get("metadatas"))
            embeddings = _as_list(result.get("embeddings"))

            for i, record_id in enumerate(ids):
                row = {
                    "id": record_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) and metas[i] is not None else {},
                }
                if include_embeddings:
                    row["embedding"] = self._normalize_embedding(embeddings[i] if i < len(embeddings) else None)
                rows_by_id[str(record_id)] = row

        return [rows_by_id[doc_id] for doc_id in normalized_ids if doc_id in rows_by_id]

    def search_by_document_ids(
        self,
        document_ids: List[str],
        *,
        limit_per_document: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve documents by metadata field `document_id`.

        One `document_id` can map to multiple vector rows (e.g., many chunk_ids).
        """
        if not self.client or not self.collection:
            logger.warning("Vector DB is not initialized; cannot query by document_id.")
            return []
        if not document_ids:
            return []

        self._ensure_collection()

        normalized_ids: List[str] = []
        seen = set()
        for x in document_ids:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            normalized_ids.append(s)
        if not normalized_ids:
            return []

        docs_by_id: Dict[str, Document] = {}
        order_map = {did: i for i, did in enumerate(normalized_ids)}
        for did in normalized_ids:
            try:
                kwargs: Dict[str, Any] = {
                    "where": {"document_id": {"$eq": did}},
                    "include": ["documents", "metadatas"],
                }
                if isinstance(limit_per_document, int) and limit_per_document > 0:
                    kwargs["limit"] = int(limit_per_document)
                result = self.collection.get(**kwargs)
                if not result or not result.get("ids"):
                    continue
                for i, row_id in enumerate(result["ids"]):
                    docs_by_id[str(row_id)] = Document(
                        id=row_id,
                        content=result["documents"][i],
                        metadata=result["metadatas"][i] or {},
                    )
            except Exception as e:
                logger.exception("Metadata document_id lookup failed for %s: %s", did, str(e))

        out = list(docs_by_id.values())
        out.sort(
            key=lambda d: (
                order_map.get(str((d.metadata or {}).get("document_id", "")), 10**9),
                str((d.metadata or {}).get("chunk_id") or d.id),
            )
        )
        return out

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
                "embedding_model": getattr(self.config.embedding, "model_name", ""),
                "embedding_provider": getattr(self.config.embedding, "provider", ""),
                "embedding_dim": getattr(self.config.embedding, "dimensions", None),
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
