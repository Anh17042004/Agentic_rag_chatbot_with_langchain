# nhập các thư viện cơ bản
from typing import Dict, List, Optional, Tuple

# nhập thư viện langchain
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

# nhập cấu hình
from config import MILVUS_URI, MILVUS_COLLECTION, EMBEDDING_MODEL

class EmbeddingManager:
    """Lớp quản lý việc tạo embedding và tương tác với vector store."""
    
    def __init__(self):
        """Khởi tạo EmbeddingManager."""
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self) -> Milvus:
        """
        Tạo kết nối đến vector store.
        
        Returns:
            Đối tượng Milvus vector store
        """
        return Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=MILVUS_COLLECTION
        )
    
    def add_documents(
        self,
        documents_dict: Dict[str, List[Document]],
        clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        Thêm tài liệu vào vector store.
        
        Args:
            documents_dict: Từ điển với tên file và danh sách các đoạn văn bản
            clear_existing: Nếu True, sẽ xóa tất cả dữ liệu cũ trước khi thêm dữ liệu mới
            
        Returns:
            Từ điển với tên file và số lượng chunks đã được thêm
        """
        results = {}
        total_chunks = 0
        is_first = True
        
        for file_name, docs in documents_dict.items():
            # Xóa dữ liệu cũ nếu là file đầu tiên và clear_existing=True
            drop_old = (is_first and clear_existing)
            
            # Thêm tài liệu vào vector store
            if docs:
                Milvus.from_documents(
                    docs,
                    self.embeddings,
                    connection_args={"uri": MILVUS_URI},
                    collection_name=MILVUS_COLLECTION,
                    drop_old=drop_old
                )
                
                # Cập nhật kết quả
                results[file_name] = len(docs)
                total_chunks += len(docs)
            else:
                results[file_name] = 0
            
            is_first = False
        
        # Cập nhật vector store
        self.vector_store = self._create_vector_store()
        
        # Thêm tổng số chunks vào kết quả
        results["total_chunks"] = total_chunks
        
        return results
    
    def similarity_search(self, query: str, k: int = 4) -> Tuple[str, List[Document]]:
        """
        Thực hiện tìm kiếm tương tự dựa trên truy vấn.
        
        Args:
            query: Truy vấn cần tìm kiếm
            k: Số lượng kết quả trả về
            
        Returns:
            Tuple gồm chuỗi kết quả đã được định dạng và danh sách các tài liệu tìm thấy
        """
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        
        return serialized, retrieved_docs
    
    def clear_vector_store(self) -> bool:
        """
        Xóa tất cả dữ liệu trong vector store.
        
        Returns:
            True nếu xóa thành công
        """
        # Xóa collection cũ và tạo collection mới
        Milvus.from_documents(
            [Document(page_content="Placeholder document", metadata={"source": "placeholder"})],
            self.embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=MILVUS_COLLECTION,
            drop_old=True
        )
        
        # Xóa document placeholder
        Milvus.from_documents(
            [],
            self.embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=MILVUS_COLLECTION,
            drop_old=True
        )
        
        # Cập nhật vector store
        self.vector_store = self._create_vector_store()
        
        return True
