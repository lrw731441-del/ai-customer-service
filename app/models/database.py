from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, create_engine, Session

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def get_session():
    with Session(engine) as session:
        yield session


def init_db():
    SQLModel.metadata.create_all(engine)


# --- 对话消息表 ---
class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str  # user / assistant
    content: str
    intent: Optional[str] = Field(default=None)
    emotion: Optional[str] = Field(default=None)
    ticket_id: Optional[int] = Field(default=None, index=True)
    response_time_ms: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- 工单表 ---
class Ticket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    user_message: str
    intent: str
    emotion: str
    priority: str  # low / medium / high / urgent
    status: str = Field(default="pending")  # pending / processing / resolved / closed
    ai_reply_draft: str
    final_reply: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(default=None)


# --- 操作日志表 ---
class AuditLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    action: str  # chat_submit / ticket_create / knowledge_upload / api_error / etc
    ip_address: str
    user_agent: Optional[str] = Field(default=None)
    detail: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- 反馈表 ---
class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int
    rating: str  # good / bad
    comment: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- 管理员用户表 ---
class AdminUser(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    password_hash: str
    role: str  # agent / admin
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- 知识库文件表 ---
class KnowledgeDocument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    file_type: str  # pdf / docx / txt / md
    file_size: int
    chunk_count: int = Field(default=0)
    status: str = Field(default="processing")  # processing / ready / error
    created_at: datetime = Field(default_factory=datetime.utcnow)
