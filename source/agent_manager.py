# nhập các thư viện cơ bản
from typing import Dict, List, Any

# nhập thư viện langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# nhập các module tùy chỉnh
from embedding_manager import EmbeddingManager
from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_TEMPLATE

class AgentManager:
    """Lớp quản lý việc tạo và sử dụng agent."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Khởi tạo AgentManager.
        
        Args:
            embedding_manager: Đối tượng EmbeddingManager để truy xuất thông tin
        """
        self.embedding_manager = embedding_manager
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()
    
    def _create_llm(self):
        """
        Tạo mô hình ngôn ngữ lớn.
        
        Returns:
            Đối tượng LLM
        """
        return ChatXAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
    
    def _create_tools(self):
        """
        Tạo các công cụ cho agent.
        
        Returns:
            Danh sách các công cụ
        """
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Truy xuất thông tin liên quan đến truy vấn."""
            return self.embedding_manager.similarity_search(query, k=2)
        
        return [retrieve]
    
    def _create_prompt(self):
        """
        Tạo prompt cho agent.
        
        Returns:
            Đối tượng ChatPromptTemplate
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    
    def _create_agent(self):
        """
        Tạo agent.
        
        Returns:
            Đối tượng Agent
        """
        return create_tool_calling_agent(self.llm, self.tools, self.prompt)
    
    def _create_agent_executor(self):
        """
        Tạo trình thực thi agent.
        
        Returns:
            Đối tượng AgentExecutor
        """
        return AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def invoke(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Gọi agent để trả lời truy vấn.
        
        Args:
            query: Truy vấn cần trả lời
            chat_history: Lịch sử trò chuyện
            
        Returns:
            Kết quả từ agent
        """
        if chat_history is None:
            chat_history = []
        
        return self.agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
