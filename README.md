# Agentic RAG (Truy xuất Tăng cường Sinh) với LangChain và Milvus

## Giới thiệu

Dự án này triển khai một hệ thống RAG (Retrieval Augmented Generation) sử dụng LangChain và Milvus. Hệ thống cho phép người dùng tải lên nhiều loại tài liệu, xử lý chúng, và đặt câu hỏi về nội dung của các tài liệu đó.

## Cấu trúc dự án

Dự án được tổ chức thành các thư mục sau:

- **modular/**: Chứa phiên bản cải tiến hoàn thiện hơn của dự án
  - `config.py`: Quản lý cấu hình và biến môi trường
  - `document_loader.py`: Xử lý việc tải và phân đoạn tài liệu
  - `embedding_manager.py`: Quản lý việc tạo embedding và tương tác với vector store
  - `agent_manager.py`: Quản lý việc tạo và sử dụng agent
  - `app.py`: Giao diện Streamlit

## Yêu cầu tiên quyết

- Python 3.11+
- tải Docker desktop về máy
https://www.docker.com/products/docker-desktop/


## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv venv
```

### 2. Kích hoạt môi trường ảo

```bash
# Windows
.\venv\Scripts\aactivate

# macOS/Linux
source venv/bin/activate
```

### 3. Cài đặt các thư viện

```bash
pip install -r requirements.txt
```

### 4. Tạo khóa API cho OpenAI

- Tạo khóa API cho OpenAI: https://platform.openai.com/api-keys

### 5. Khởi động Milvus bằng Docker (cần mở docker desktop trước)

```bash
docker-compose up --build
```

Bạn có thể truy cập giao diện web của Milvus tại: http://localhost:9091/


- Mở file `.env` và thêm khóa API của bạn:

```
OPENAI_API_KEY = "sk-your-api-key-here"
XAI_API_KEY = "sk-your-api-key-here"
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "documents"
```

## Thực thi ứng dụng

Chạy ứng dụng Streamlit (phiên bản module hóa):

- Truy cập vào package "modular" trong thư mục gốc của dự án:
- Chạy ứng dụng Streamlit:

```bash
streamlit run app.py
```

## Tính năng

### Tải lên và xử lý tài liệu

- Hỗ trợ nhiều định dạng file: PDF, Word, Excel, PowerPoint, CSV, HTML, Markdown, và văn bản thông thường
- Tùy chỉnh kích thước đoạn văn bản và độ chồng lập
- Xóa dữ liệu cũ trước khi thêm dữ liệu mới (tùy chọn)

### Truy vấn thông tin

- Đặt câu hỏi về nội dung của các tài liệu đã tải lên
- Trả lời dựa trên thông tin từ các tài liệu
- Trích dẫn nguồn thông tin

### Quản lý tài liệu và trò chuyện

- Xóa tất cả tài liệu đã tải lên
- Xóa lịch sử trò chuyện

## Cách sử dụng

1. Khởi động ứng dụng bằng lệnh `streamlit run modular/app.py`
2. Trong sidebar bên trái, chọn tab "Tải lên"
3. Tải lên một hoặc nhiều tài liệu (PDF, Word, Excel, v.v.)
4. Tùy chỉnh các tham số xử lý trong phần "Tùy chọn nâng cao" (nếu cần)
5. Chọn có hoặc không xóa dữ liệu cũ trong phần "Tùy chọn xử lý"
6. Nhấn nút "Xử lý tài liệu" và đợi quá trình hoàn tất
7. Đặt câu hỏi về nội dung của các tài liệu trong thanh chat ở dưới cùng
8. Để quản lý tài liệu hoặc xóa lịch sử trò chuyện, sử dụng tab "Quản lý" trong sidebar

## Nguồn tham khảo

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Milvus Documentation](https://milvus.io/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
