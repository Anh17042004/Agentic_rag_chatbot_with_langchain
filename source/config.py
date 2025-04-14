# nhập các thư viện cơ bản
import os
from dotenv import load_dotenv

# tải các biến môi trường
load_dotenv()

# Cấu hình Milvus
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "documents")

# Cấu hình OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Cấu hình mặc định cho xử lý tài liệu
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Cấu hình mô hình
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "grok-3-mini-beta"
LLM_TEMPERATURE = 0.1

# Cấu hình prompt
SYSTEM_TEMPLATE = """Bạn là trợ lý AI thông minh được tạo bởi Anh Đoàn Tuấn Anh dzaidzai, hữu ích và chính xác.
Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên tài liệu được cung cấp.
Khi bạn không biết câu trả lời, hãy đưa ra thông báo rằng không tìm thấy tài liệu.
Luôn trích dẫn nguồn thông tin khi bạn sử dụng dữ liệu từ tài liệu.
Trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu.
"""
