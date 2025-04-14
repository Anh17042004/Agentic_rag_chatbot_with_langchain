# nhập các thư viện cơ bản
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time

# nhập các module tùy chỉnh
from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from agent_manager import AgentManager
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Thiết lập trang Streamlit
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜", layout="wide")
st.title("🦜 Agentic RAG Chatbot")

# Khởi tạo các đối tượng quản lý
@st.cache_resource
def get_managers():
    """
    Khởi tạo và lưu cache các đối tượng quản lý.
    
    Returns:
        Tuple gồm EmbeddingManager và AgentManager
    """
    embedding_manager = EmbeddingManager()
    agent_manager = AgentManager(embedding_manager)
    return embedding_manager, agent_manager

# Lấy các đối tượng quản lý
embedding_manager, agent_manager = get_managers()


# Tạo sidebar cho việc tải lên tài liệu
with st.sidebar:
    st.header("Quản lý tài liệu")
    
    # Tạo tab cho các chức năng quản lý tài liệu
    upload_tab, manage_tab = st.tabs(["Tải lên", "Quản lý"])
    
    with upload_tab:
        st.subheader("Tải lên tài liệu mới")
        
        # Thêm hướng dẫn
        st.markdown("""
        Hỗ trợ các định dạng file:
        - PDF (.pdf)
        - Văn bản (.txt)
        - Excel (.xlsx, .xls)
        - Word (.docx, .doc)
        - PowerPoint (.pptx, .ppt)
        - CSV (.csv)
        - HTML (.html)
        - Markdown (.md)
        """)
        
        # Tạo widget tải lên nhiều file
        uploaded_files = st.file_uploader(
            "Chọn file tài liệu",
            accept_multiple_files=True,
            type=["pdf", "txt", "csv", "docx", "doc", "pptx", "ppt", "html", "md", "xlsx", "xls"],
            help="Tải lên một hoặc nhiều file tài liệu để xử lý"
        )
        
        # Tạo các tùy chọn nâng cao
        with st.expander("Tùy chọn nâng cao"):
            chunk_size = st.slider(
                "Kích thước đoạn văn bản (chunk size)",
                min_value=100,
                max_value=2000,
                value=DEFAULT_CHUNK_SIZE,
                step=100,
                help="Kích thước của mỗi đoạn văn bản sau khi chia nhỏ"
            )
            
            chunk_overlap = st.slider(
                "Độ chồng lập giữa các đoạn (chunk overlap)",
                min_value=0,
                max_value=500,
                value=DEFAULT_CHUNK_OVERLAP,
                step=50,
                help="Số ký tự chồng lập giữa các đoạn liên tiếp"
            )
        
        # Tùy chọn xóa dữ liệu cũ
        if uploaded_files:
            with st.expander("Tùy chọn xử lý", expanded=True):
                clear_existing = st.checkbox(
                    "Xóa tất cả dữ liệu cũ trước khi thêm dữ liệu mới",
                    value=False,
                    help="Nếu được chọn, tất cả dữ liệu cũ sẽ bị xóa trước khi thêm dữ liệu mới"
                )
                
                # Nút xử lý tài liệu
                if st.button("Xử lý tài liệu", type="primary"):
                    with st.spinner("Đang xử lý tài liệu..."):
                        try:
                            # Tải và phân đoạn tài liệu
                            documents_dict = DocumentLoader.load_and_split_documents(
                                uploaded_files,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            
                            # Thêm tài liệu vào vector store
                            results = embedding_manager.add_documents(
                                documents_dict,
                                clear_existing=clear_existing
                            )
                            
                            # Lấy tổng số chunks
                            total_chunks = results.pop("total_chunks", 0)
                            
                            # Hiển thị kết quả
                            st.success(f"Đã xử lý {len(uploaded_files)} file thành công!")
                            
                            # Hiển thị thông tin chi tiết
                            st.write(f"**Tổng số đoạn văn bản đã tạo:** {total_chunks}")
                            
                            # Hiển thị thông tin từng file
                            st.write("**Chi tiết từng file:**")
                            for file_name, num_chunks in results.items():
                                st.write(f"- {file_name}: {num_chunks} đoạn văn bản")
                            
                            # Thêm thông báo thành công
                            st.session_state.upload_success = True
                            
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý tài liệu: {str(e)}")
    
    with manage_tab:
        st.subheader("Quản lý tài liệu và trò chuyện")
        
        # Tạo 2 cột để phân chia giao diện
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Xóa tất cả tài liệu**")
            
            # Khởi tạo session state cho checkbox xóa tài liệu
            if "confirm_delete_docs" not in st.session_state:
                st.session_state.confirm_delete_docs = False
            
            # Checkbox xác nhận xóa tài liệu
            st.session_state.confirm_delete_docs = st.checkbox(
                "Xác nhận xóa tất cả tài liệu",
                value=st.session_state.confirm_delete_docs,
                key="delete_docs_checkbox"
            )
            
            # Nút xóa tất cả tài liệu
            if st.button(
                "Xóa tài liệu",
                type="primary",
                disabled=not st.session_state.confirm_delete_docs,
                use_container_width=True
            ):
                with st.spinner("Đang xóa tất cả tài liệu..."):
                    try:
                        embedding_manager.clear_vector_store()
                        st.success("Đã xóa thành công tất cả tài liệu!")
                        st.info("Hệ thống đã được làm mới. Bạn có thể tải lên tài liệu mới.")
                        st.session_state.confirm_delete_docs = False
                    except Exception as e:
                        st.error(f"Lỗi khi xóa tài liệu: {str(e)}")
        
        with col2:
            st.write("**Xóa lịch sử trò chuyện**")
            
            # Khởi tạo session state cho checkbox xóa lịch sử
            if "confirm_delete_history" not in st.session_state:
                st.session_state.confirm_delete_history = False
            
            # Checkbox xác nhận xóa lịch sử
            st.session_state.confirm_delete_history = st.checkbox(
                "Xác nhận xóa lịch sử trò chuyện",
                value=st.session_state.confirm_delete_history,
                key="delete_history_checkbox"
            )
            
            # Nút xóa lịch sử trò chuyện
            if st.button(
                "Xóa lịch sử",
                type="primary",
                disabled=not st.session_state.confirm_delete_history,
                use_container_width=True
            ):
                with st.spinner("Đang xóa lịch sử trò chuyện..."):
                    try:
                        # Xóa lịch sử trò chuyện
                        st.session_state.messages = []
                        st.success("Đã xóa thành công lịch sử trò chuyện!")
                        st.session_state.confirm_delete_history = False
                    except Exception as e:
                        st.error(f"Lỗi khi xóa lịch sử trò chuyện: {str(e)}")

# Hiển thị thông báo nếu vừa tải lên tài liệu thành công
if st.session_state.get("upload_success", False):
    st.success("Tài liệu đã được tải lên và xử lý thành công! Bạn có thể bắt đầu đặt câu hỏi.")
    # Xóa trạng thái thành công để không hiển thị lại khi tải lại trang
    st.session_state.upload_success = False

# khởi tạo lịch sử trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = []

# hiển thị tin nhắn trò chuyện từ lịch sử khi chạy lại ứng dụng
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# tạo thanh nhập liệu để nhập tin nhắn
user_question = st.chat_input("Hỏi tôi bất cứ điều gì về tài liệu của bạn...")


# người dùng đã gửi prompt chưa?
if user_question:
    # thêm tin nhắn từ người dùng (prompt) vào màn hình với streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    # gọi agent
    with st.spinner("Đang tìm kiếm thông tin..."):
        result = agent_manager.invoke(user_question, chat_history=st.session_state.messages)
        ai_message = result["output"]

    # thêm phản hồi từ llm vào màn hình (và trò chuyện)
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))



