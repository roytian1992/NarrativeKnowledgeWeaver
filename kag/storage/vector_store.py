"""
向量数据库存储模块

基于ChromaDB的向量存储和语义搜索
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ..models.entities import Document
from ..utils.config import KAGConfig


class VectorStore:
    """向量数据库存储"""
    
    def __init__(self, config: KAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化向量数据库"""
        try:
            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(
                path=self.config.storage.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name="kag_documents",
                metadata={"description": "KAG文档向量存储"}
            )
            
            # 初始化嵌入模型
            self.embedding_model = SentenceTransformer(
                self.config.storage.embedding_model_name
            )
            
            print("✅ 向量数据库初始化成功")
            
        except Exception as e:
            print(f"❌ 向量数据库初始化失败: {str(e)}")
            self.client = None
            
    def _ensure_collection(self):
        """确保 collection 可用，若被删除则自动重新获取"""
        try:
            _ = self.collection.count()
        except:
            self.collection = self.client.get_or_create_collection(
                name="kag_documents",
                metadata={"description": "KAG文档向量存储"}
            )
    
    def store_documents(self, documents: List[Document]) -> None:
        """存储文档到向量数据库"""
        if not self.client or not self.collection:
            print("⚠️ 向量数据库未初始化，跳过向量存储")
            return
        
        self._ensure_collection()  
        try:
            # 准备数据
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                ids.append(doc.id)
                texts.append(doc.content)
                
                # 准备元数据（ChromaDB要求所有值都是字符串、数字或布尔值）
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list):
                        metadata[key] = str(value)  # 转换为字符串
                    elif isinstance(value, dict):
                        metadata[key] = str(value)  # 转换为字符串
                    else:
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # 生成嵌入向量
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # 存储到ChromaDB
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
        """语义搜索"""
        if not self.client or not self.collection:
            return []
        self._ensure_collection()  
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # 转换结果
            documents = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # 添加相似度分数
                metadata["similarity_score"] = 1 - distance
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ 语义搜索失败: {str(e)}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Document]:
        """根据元数据搜索"""
        if not self.client or not self.collection:
            return []
        self._ensure_collection()  
        
        try:
            # 构建where条件
            where_conditions = {}
            for key, value in metadata_filter.items():
                if isinstance(value, dict):
                    # 用户自定义了操作符，例如 {"$in": [...]}
                    where_conditions[key] = value
                else:
                    # 默认使用等值匹配
                    where_conditions[key] = {"$eq": value}
            
            # 执行搜索
            results = self.collection.get(
                where=where_conditions,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            # 转换结果
            documents = []
            for i in range(len(results["ids"])):
                doc_id = results["ids"][i]
                content = results["documents"][i]
                metadata = results["metadatas"][i]
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ 元数据搜索失败: {str(e)}")
            return []
        

    def search_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """
        根据一组文档 id 批量获取记录
        Args:
            doc_ids: 存储时使用的 id 列表
        Returns:
            Document 对象列表（按传入顺序返回；若某个 id 未命中则忽略）
        """
        if not self.client or not self.collection:
            print("⚠️ 向量数据库未初始化，无法检索")
            return []

        if not doc_ids:
            return []
        
        self._ensure_collection()  

        try:
            result = self.collection.get(
                ids=doc_ids,
                include=["documents", "metadatas"]
            )
            documents = []
            for i, doc_id in enumerate(result["ids"]):
                # get() 仅返回命中的 id；保持顺序与返回结果一致
                content = result["documents"][i]
                metadata = result["metadatas"][i]
                documents.append(
                    Document(id=doc_id, content=content, metadata=metadata)
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
                "document_count": count,
                "embedding_model": self.config.storage.embedding_model
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def delete_collection(self) -> None:
        """删除集合"""
        if self.client and self.collection:
            try:
                self.client.delete_collection("kag_documents")
                print("✅ 向量集合已删除")
            except Exception as e:
                print(f"❌ 删除向量集合失败: {str(e)}")

