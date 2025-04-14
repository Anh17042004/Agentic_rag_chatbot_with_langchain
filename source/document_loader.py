# nhập các thư viện cơ bản
import os
import time
import tempfile
from typing import Dict, List
from pathlib import Path

# nhập thư viện langchain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# nhập cấu hình
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class DocumentLoader:
    """Lớp xử lý việc tải và phân đoạn tài liệu."""
    
    @staticmethod
    def get_loader_for_file(file_path: str):
        """
        Trả về loader phù hợp cho loại file.
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            Loader phù hợp cho loại file
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif file_extension == '.csv':
            return CSVLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            return Docx2txtLoader(file_path)
        elif file_extension in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif file_extension == '.html':
            return UnstructuredHTMLLoader(file_path)
        elif file_extension == '.md':
            return UnstructuredMarkdownLoader(file_path)
        # Đã loại bỏ xử lý file JSON
        elif file_extension in ['.xls', '.xlsx']:
            return UnstructuredExcelLoader(file_path)
        else:
            # Mặc định sử dụng TextLoader cho các loại file không xác định
            return TextLoader(file_path, encoding='utf-8')
    
    @staticmethod
    def load_and_split_documents(
        uploaded_files,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Dict[str, List[Document]]:
        """
        Tải và phân đoạn tài liệu từ các file được tải lên.
        
        Args:
            uploaded_files: Danh sách các file được tải lên qua Streamlit
            chunk_size: Kích thước của mỗi đoạn văn bản
            chunk_overlap: Độ chồng lập giữa các đoạn
            
        Returns:
            Từ điển với tên file và danh sách các đoạn văn bản đã được phân đoạn
        """
        result = {}
        
        for uploaded_file in uploaded_files:
            # Tạo thư mục tạm thời để lưu file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Lưu file tải lên vào thư mục tạm thời
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Lấy loader phù hợp cho loại file
                loader = DocumentLoader.get_loader_for_file(temp_file_path)
                
                # Tải tài liệu
                documents = loader.load()
                
                # Thêm metadata về nguồn gốc file
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["file_type"] = Path(uploaded_file.name).suffix.lower()
                    doc.metadata["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Chia tài liệu thành các đoạn nhỏ
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                docs = text_splitter.split_documents(documents)
                
                # Lưu kết quả
                result[uploaded_file.name] = docs
        
        return result
