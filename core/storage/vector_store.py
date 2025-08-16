"""
向量数据库存储模块

基于ChromaDB的向量存储和语义搜索
"""

from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings
from core.models.data import Document
from ..utils.config import KAGConfig


class VectorStore:
    """向量数据库存储"""

    def __init__(self, config: KAGConfig, category: str = "documents"):
        self.config = config
        self.vector_store_path = os.path.join(self.config.storage.vector_store_path, category)
        self.vector_store_name = category  # 集合名=类目名
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self) -> None:
        """初始化向量数据库"""
        try:
            # 初始化ChromaDB客户端（本地持久化）
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建集合（用当前类目名）
            self.collection = self.client.get_or_create_collection(
                name=self.vector_store_name,
                metadata={"description": f"{self.vector_store_name} vectordb"}
            )

            # 初始化嵌入模型
            if self.config.vectordb_embedding.provider == "openai":
                from core.model_providers.openai_embedding import OpenAIEmbeddingModel
                self.embedding_model = OpenAIEmbeddingModel(self.config.vectordb_embedding)
            else:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.config.vectordb_embedding.model_name)

            print(f"✅ 向量数据库初始化成功: path={self.vector_store_path}, collection={self.vector_store_name}")

        except Exception as e:
            print(f"❌ 向量数据库初始化失败: {str(e)}")
            self.client = None
            self.collection = None

    def _ensure_collection(self):
        """确保 collection 可用，若被删除则自动重新获取"""
        if not self.client:
            return
        try:
            _ = self.collection.count()
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=self.vector_store_name,
                metadata={"description": f"{self.vector_store_name} vectordb"}
            )

    def store_documents(self, documents: List[Document]) -> None:
        """存储文档到向量数据库"""
        if not self.client or not self.collection:
            print("⚠️ 向量数据库未初始化，跳过向量存储")
            return

        self._ensure_collection()
        try:
            ids, texts, metadatas = [], [], []

            for doc in documents:
                ids.append(str(doc.id))
                texts.append(doc.content)

                # 元数据（Chroma 仅支持标量；其它转字符串）
                metadata = {}
                for key, value in (doc.metadata or {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                metadatas.append(metadata)

            # 生成嵌入向量
            embeddings = self.embedding_model.encode(texts)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            # 存储
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"✅ 成功存储 {len(documents)} 个文档到向量数据库")

        except Exception as e:
            print(f"❌ 向量存储失败: {str(e)}")

    def search(self, query: str, limit: int = 5) -> List[Document]:
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
                include=["documents", "metadatas", "distances"]  # ✅ 去掉 "ids"
            )

            documents: List[Document] = []
            if not results or not results.get("ids"):
                return documents

            ids = results["ids"][0]                     # ← 仍然可用（默认返回）
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results.get("distances", [[None]*len(ids)])[0]

            for i in range(len(ids)):
                meta = metas[i] or {}
                if dists[i] is not None:
                    meta["similarity_score"] = 1.0 - float(dists[i])
                documents.append(Document(id=ids[i], content=docs[i], metadata=meta))
            return documents
        except Exception as e:
            print(f"❌ 语义搜索失败: {str(e)}")
            return []


    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Document]:
        if not self.client or not self.collection:
            return []
        self._ensure_collection()
        try:
            where_conditions = {}
            for key, value in (metadata_filter or {}).items():
                if isinstance(value, dict):
                    where_conditions[key] = value
                else:
                    where_conditions[key] = {"$eq": value}

            results = self.collection.get(
                where=where_conditions,
                limit=limit,
                include=["documents", "metadatas"]  # ✅ 去掉 "ids"
            )

            documents: List[Document] = []
            if not results or not results.get("ids"):
                return documents

            for i, doc_id in enumerate(results["ids"]):  # ← 仍然可用（默认返回）
                documents.append(
                    Document(
                        id=doc_id,
                        content=results["documents"][i],
                        metadata=results["metadatas"][i] or {}
                    )
                )
            return documents
        except Exception as e:
            print(f"❌ 元数据搜索失败: {str(e)}")
            return []


    def search_by_ids(self, doc_ids: List[str]) -> List[Document]:
        if not self.client or not self.collection:
            print("⚠️ 向量数据库未初始化，无法检索")
            return []
        if not doc_ids:
            return []

        self._ensure_collection()
        try:
            result = self.collection.get(
                ids=[str(x) for x in doc_ids],
                include=["documents", "metadatas"]  # ✅ 去掉 "ids"
            )
            documents: List[Document] = []
            if not result or not result.get("ids"):
                return documents

            for i, doc_id in enumerate(result["ids"]):  # ← 仍然可用（默认返回）
                documents.append(
                    Document(
                        id=doc_id,
                        content=result["documents"][i],
                        metadata=result["metadatas"][i] or {}
                    )
                )
            return documents
        except Exception as e:
            print(f"❌ 批量 ID 检索失败: {str(e)}")
            return []


    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.client or not self.collection:
            return {"status": "disconnected"}

        self._ensure_collection()

        try:
            count = self.collection.count()
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
            return {"status": "error", "error": str(e)}

    def delete_collection(self) -> None:
        """删除集合"""
        if self.client and self.collection:
            try:
                self.client.delete_collection(self.vector_store_name)
                print(f"✅ 向量集合已删除: {self.vector_store_name}")
                # 重新创建同名空集合，保证实例可继续使用
                self.collection = self.client.get_or_create_collection(
                    name=self.vector_store_name,
                    metadata={"description": f"{self.vector_store_name} vectordb"}
                )
            except Exception as e:
                print(f"❌ 删除向量集合失败: {str(e)}")
