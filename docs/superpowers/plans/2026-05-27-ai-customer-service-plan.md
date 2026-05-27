# AI 客服工单智能处理系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 FastAPI + LangGraph + RAG + DeepSeek 的智能客服工单处理系统，支持对话优先的自动回复与工单兜底路由。

**Architecture:** FastAPI 单进程服务，LangGraph 3 节点 Agent（意图+情绪 → RAG检索 → 回复+路由），SQLite + ChromaDB 嵌入模式本地存储，HTMX + Alpine.js + Chart.js 前端，Docker 单容器部署。

**Tech Stack:** Python 3.12, FastAPI, LangGraph, DeepSeek API, ChromaDB, SQLModel, HTMX, Alpine.js, Chart.js

---

## 文件结构

```
ai-customer-service/
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口 + StaticFiles + lifespan
│   ├── config.py            # 环境变量配置
│   ├── dependencies.py      # Token鉴权、IP限流依赖
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # POST /api/v1/chat
│   │   ├── tickets.py       # GET/PATCH /api/v1/tickets
│   │   ├── knowledge.py     # POST/GET/DELETE /api/v1/knowledge
│   │   ├── dashboard.py     # GET /api/v1/dashboard/stats
│   │   └── auth.py          # POST /api/v1/auth/login, GET /api/v1/auth/token
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph StateGraph 定义
│   │   ├── nodes.py         # 3 个节点函数实现
│   │   └── prompts.py       # Prompt 模板常量
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectordb.py      # ChromaDB 集合管理 + 检索
│   │   ├── loader.py        # 文档解析（PDF/DOCX/TXT/MD）
│   │   └── splitter.py      # 语义分块
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py      # SQLModel 表定义 + init_db
│   └── services/
│       ├── __init__.py
│       ├── chat_service.py  # 对话编排逻辑
│       └── stats_service.py # 看板统计查询
├── frontend/
│   ├── index.html           # 对话页
│   ├── login.html           # 登录页
│   ├── admin.html           # 审核台
│   └── dashboard.html       # 看板
├── knowledge_base/          # 4 个 Globex 知识库 .md 文件（任务中生成）
├── data/                    # 运行时自动创建
├── tests/
│   ├── __init__.py
│   ├── test_chat.py
│   └── test_tickets.py
├── evaluate.py
├── seed_data.py
├── Dockerfile
└── README.md
```

---

## Sprint 1：核心后端跑通

### Task 1: 项目骨架搭建

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`

- [ ] **Step 1: 创建 requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlmodel==0.0.22
chromadb==0.5.23
langgraph==0.2.60
langchain-core==0.3.29
openai==1.59.3
python-multipart==0.0.19
python-dotenv==1.0.1
pypdf==5.1.0
python-docx==1.1.2
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
httpx==0.28.1
pytest==8.3.4
pytest-asyncio==0.25.0
```

- [ ] **Step 2: 创建 .env.example**

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DATABASE_URL=sqlite:///./data/tickets.db
CHROMA_PERSIST_DIR=./data/chroma
HOST=0.0.0.0
PORT=8000
API_TOKEN_SECRET=change-me-to-a-random-string
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=  # 用 scripts/hash_password.py 生成
RATE_LIMIT_PER_MINUTE=30
```

- [ ] **Step 3: 创建 app/config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/tickets.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_TOKEN_SECRET = os.getenv("API_TOKEN_SECRET", "dev-secret")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
```

- [ ] **Step 4: 创建 app/main.py（最小骨架）**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import HOST, PORT

app = FastAPI(title="AI客服工单智能处理系统", version="1.0.0")

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
```

- [ ] **Step 5: 安装依赖并验证启动**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 浏览器打开 http://localhost:8000/docs 应看到 Swagger 页面
```

- [ ] **Step 6: 创建 data/ 和 knowledge_base/ 目录**

```bash
mkdir -p data knowledge_base
```

---

### Task 2: 数据库模型

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/database.py`

- [ ] **Step 1: 创建 app/models/__init__.py**

```python
```

- [ ] **Step 2: 创建 app/models/database.py**

```python
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
```

- [ ] **Step 3: 注册 lifespan 到 main.py**

更新 `app/main.py`，添加数据库初始化：

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import HOST, PORT
from app.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI客服工单智能处理系统", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")
```

- [ ] **Step 4: 验证数据库初始化**

```bash
python -c "from app.models.database import init_db; init_db(); print('DB OK')"
# 应输出 DB OK，且 data/ 下出现 tickets.db
```

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: add project scaffolding and database models"
```

---

### Task 3: Agent 核心 — Prompt 模板 + 节点实现

**Files:**
- Create: `app/agent/__init__.py`
- Create: `app/agent/prompts.py`
- Create: `app/agent/nodes.py`

- [ ] **Step 1: 创建 app/agent/__init__.py**

```python
```

- [ ] **Step 2: 创建 app/agent/prompts.py**

```python
INTENT_EMOTION_PROMPT = """你是一个专业的客服分析助手。请分析以下用户消息，输出意图和情绪。

意图选项（选一个）：咨询、投诉、退款、售后、要求转人工、其他
情绪选项（选一个）：平静、不满、愤怒、紧急

用户消息：{user_message}

请严格输出以下 JSON 格式，不要输出其他内容：
{{"intent": "咨询", "emotion": "平静"}}"""

REPLY_ROUTING_PROMPT = """你是跨境出海电商平台「Globex 全球购」的专业客服。请根据以下信息生成回复。

## 用户信息
- 用户消息：{user_message}
- 用户意图：{intent}
- 用户情绪：{emotion}

## 对话历史（最近 5 条）
{chat_history}

## 知识库检索结果
{retrieved_docs}

## 回复要求
1. 如果知识库有相关内容，基于知识库内容生成专业、友好、具体的回复
2. 如果知识库无相关内容，生成引导性回复，告知用户将创建工单由人工客服处理
3. 如果用户情绪愤怒或紧急，优先表达理解和歉意，安抚用户情绪
4. 如果用户意图是"要求转人工"，直接告知用户将转接人工客服
5. 回复语气专业但温暖，不要使用模板化的"亲"等过于随意的称呼

## 请输出以下 JSON 格式，不要输出其他内容：
{{
  "reply": "这里写生成的回复内容",
  "create_ticket": true或false,
  "ticket_priority": "low/medium/high/urgent"
}}"""

FALLBACK_REPLY = "抱歉，系统繁忙，请稍后重试。如需紧急帮助，请拨打客服热线 400-888-0000。"
```

- [ ] **Step 3: 创建 app/agent/nodes.py**

```python
import json
import time
from typing import List, Dict, Any
from openai import OpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from app.agent.prompts import INTENT_EMOTION_PROMPT, REPLY_ROUTING_PROMPT, FALLBACK_REPLY

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def detect_intent_and_emotion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: 意图识别 + 情绪检测"""
    user_message = state["user_message"]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": INTENT_EMOTION_PROMPT.format(user_message=user_message)}],
            temperature=0.1,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "intent": result.get("intent", "其他"),
            "emotion": result.get("emotion", "平静"),
        }
    except Exception:
        return {"intent": "其他", "emotion": "平静"}


def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: RAG 检索"""
    from app.rag.vectordb import search_similar

    user_message = state["user_message"]
    docs = search_similar(user_message, top_k=5)

    return {"retrieved_docs": docs}


def generate_reply_and_route(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3: 回复生成 + 路由决策"""
    user_message = state["user_message"]
    intent = state.get("intent", "其他")
    emotion = state.get("emotion", "平静")
    retrieved_docs = state.get("retrieved_docs", [])
    chat_history = state.get("chat_history", "")

    # 格式化检索结果
    if retrieved_docs:
        docs_text = "\n\n---\n\n".join(
            [f"【来源：{d.get('source', '未知')}】\n{d.get('content', '')}" for d in retrieved_docs]
        )
    else:
        docs_text = "（知识库中暂无相关内容）"

    # 愤怒/紧急/转人工 → 直接建单，不需要检索结果
    if emotion in ("愤怒", "紧急") or intent == "要求转人工":
        priority = "urgent" if emotion == "紧急" else "high"
        reply = "非常抱歉给您带来不便，我已收到您的反馈，正在为您创建优先处理工单，客服专员将尽快与您联系。"

        start = time.time()
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": REPLY_ROUTING_PROMPT.format(
                        user_message=user_message,
                        intent=intent,
                        emotion=emotion,
                        chat_history=chat_history,
                        retrieved_docs=docs_text,
                    )}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            result = json.loads(response.choices[0].message.content.strip())
            reply = result.get("reply", reply)
        except Exception:
            pass

        return {
            "ai_reply": reply,
            "create_ticket": True,
            "ticket_priority": priority,
        }

    # 正常流程
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": REPLY_ROUTING_PROMPT.format(
                    user_message=user_message,
                    intent=intent,
                    emotion=emotion,
                    chat_history=chat_history,
                    retrieved_docs=docs_text,
                )}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "ai_reply": result.get("reply", FALLBACK_REPLY),
            "create_ticket": result.get("create_ticket", False),
            "ticket_priority": result.get("ticket_priority", "low"),
        }
    except Exception:
        return {
            "ai_reply": FALLBACK_REPLY,
            "create_ticket": True,
            "ticket_priority": "medium",
        }
```

- [ ] **Step 4: 提交**

```bash
git add app/agent/
git commit -m "feat: add agent prompt templates and 3 node functions"
```

---

### Task 4: LangGraph 状态图定义

**Files:**
- Create: `app/agent/graph.py`

- [ ] **Step 1: 创建 app/agent/graph.py**

```python
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.nodes import detect_intent_and_emotion, retrieve_knowledge, generate_reply_and_route


class AgentState(dict):
    """Agent 状态，继承 dict 兼容 LangGraph"""
    user_message: str = ""
    chat_history: str = ""
    intent: str = ""
    emotion: str = ""
    retrieved_docs: list = []
    ai_reply: str = ""
    create_ticket: bool = False
    ticket_priority: str = "low"


def should_retrieve(state: Dict[str, Any]) -> str:
    """条件路由：愤怒/紧急/转人工 → 跳过 RAG"""
    if state.get("emotion") in ("愤怒", "紧急") or state.get("intent") == "要求转人工":
        return "reply_routing"
    return "rag_retrieval"


def build_graph() -> StateGraph:
    workflow = StateGraph(dict)

    workflow.add_node("intent_emotion", detect_intent_and_emotion)
    workflow.add_node("rag_retrieval", retrieve_knowledge)
    workflow.add_node("reply_routing", generate_reply_and_route)

    workflow.set_entry_point("intent_emotion")

    workflow.add_conditional_edges("intent_emotion", should_retrieve, {
        "rag_retrieval": "rag_retrieval",
        "reply_routing": "reply_routing",
    })
    workflow.add_edge("rag_retrieval", "reply_routing")
    workflow.add_edge("reply_routing", END)

    return workflow.compile()


agent_graph = build_graph()


def run_agent(user_message: str, chat_history: str = "") -> Dict[str, Any]:
    """执行 Agent 并返回结果"""
    result = agent_graph.invoke({
        "user_message": user_message,
        "chat_history": chat_history,
    })
    return {
        "intent": result.get("intent", "其他"),
        "emotion": result.get("emotion", "平静"),
        "ai_reply": result.get("ai_reply", ""),
        "create_ticket": result.get("create_ticket", False),
        "ticket_priority": result.get("ticket_priority", "low"),
    }
```

- [ ] **Step 2: 验证 Agent 图可以构建**

```bash
python -c "from app.agent.graph import build_graph; g = build_graph(); print('Graph OK:', type(g).__name__)"
```

- [ ] **Step 3: 提交**

```bash
git add app/agent/graph.py
git commit -m "feat: add LangGraph state graph with conditional routing"
```

---

### Task 5: RAG 模块 — 文档加载 + 语义切分 + ChromaDB

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/loader.py`
- Create: `app/rag/splitter.py`
- Create: `app/rag/vectordb.py`

- [ ] **Step 1: 创建 app/rag/__init__.py**

```python
```

- [ ] **Step 2: 创建 app/rag/loader.py**

```python
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_text_file(filepath: str) -> str:
    """加载纯文本文件 (txt/md)"""
    return Path(filepath).read_text(encoding="utf-8")


def load_pdf(filepath: str) -> str:
    """加载 PDF 文件"""
    reader = PdfReader(filepath)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def load_docx(filepath: str) -> str:
    """加载 Word 文件"""
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])


def load_document(filepath: str, file_type: str) -> str:
    """根据文件类型加载文档"""
    loaders = {
        "txt": load_text_file,
        "md": load_text_file,
        "pdf": load_pdf,
        "docx": load_docx,
    }
    loader = loaders.get(file_type)
    if not loader:
        raise ValueError(f"不支持的文件类型: {file_type}")
    return loader(filepath)
```

- [ ] **Step 3: 创建 app/rag/splitter.py**

```python
import re
from typing import List, Dict


def semantic_split(text: str, max_chunk_size: int = 500, source: str = "") -> List[Dict[str, str]]:
    """语义切分：按段落+句子边界切分，保持语义完整"""

    # 先按段落切分
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果加入当前段落不超限，合并
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n"
        else:
            # 当前 chunk 够大了，保存
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    # 保存最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [{"content": chunk, "source": source} for chunk in chunks]
```

- [ ] **Step 4: 创建 app/rag/vectordb.py**

```python
import uuid
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from app.config import CHROMA_PERSIST_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

_chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

COLLECTION_NAME = "globex_knowledge"


def get_collection():
    """获取或创建知识库集合"""
    return _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def embed_text(text: str) -> List[float]:
    """调用 DeepSeek Embedding"""
    response = client.embeddings.create(
        model="deepseek-chat",
        input=text,
    )
    return response.data[0].embedding


def add_documents(chunks: List[Dict[str, str]]) -> int:
    """将文档片段存入 ChromaDB，返回片段数"""
    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [{"source": c.get("source", "")} for c in chunks]
    embeddings = [embed_text(doc) for doc in documents]

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)


def search_similar(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """向量检索，返回 Top-K 相似文档片段"""
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            source = ""
            if results["metadatas"] and results["metadatas"][0] and i < len(results["metadatas"][0]):
                source = results["metadatas"][0][i].get("source", "")
            distance = 0
            if results["distances"] and results["distances"][0] and i < len(results["distances"][0]):
                distance = results["distances"][0][i]
            similarity = 1 - distance if distance else 0
            if similarity >= 0.7:
                docs.append({"content": doc, "source": source, "similarity": similarity})

    return docs


def delete_collection():
    """删除知识库集合（用于重新上传）"""
    try:
        _chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def get_document_count() -> int:
    """获取知识库文档片段数"""
    collection = get_collection()
    return collection.count()
```

- [ ] **Step 5: 验证 RAG 模块导入无误**

```bash
python -c "from app.rag.loader import load_document; from app.rag.splitter import semantic_split; from app.rag.vectordb import get_collection; print('RAG module OK')"
```

- [ ] **Step 6: 提交**

```bash
git add app/rag/
git commit -m "feat: add RAG module - loader, splitter, chromadb vector store"
```

---

### Task 6: API — Auth 鉴权（Token + Login）

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/dependencies.py`
- Create: `app/api/auth.py`

- [ ] **Step 1: 创建 app/api/__init__.py**

```python
```

- [ ] **Step 2: 创建 app/dependencies.py**

```python
import time
from collections import defaultdict
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.config import API_TOKEN_SECRET, RATE_LIMIT_PER_MINUTE

security = HTTPBearer(auto_error=False)

# 简易内存限流
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def rate_limit(request: Request):
    """IP 限流：每 IP 每分钟最多 N 次请求"""
    now = time.time()
    ip = request.client.host if request.client else "unknown"

    # 清理 1 分钟前的记录
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < 60]

    if len(_rate_limit_store[ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    _rate_limit_store[ip].append(now)


def verify_session_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 Session Token（对话页用）"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少认证 Token")
    try:
        payload = jwt.decode(credentials.credentials, API_TOKEN_SECRET, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 JWT Token（管理端用）"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="请先登录")
    try:
        payload = jwt.decode(credentials.credentials, API_TOKEN_SECRET, algorithms=["HS256"])
        if payload.get("type") != "admin":
            raise HTTPException(status_code=403, detail="权限不足")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")
```

- [ ] **Step 3: 创建 app/api/auth.py**

```python
import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlmodel import Session, select
from passlib.hash import bcrypt
from jose import jwt
from datetime import datetime, timedelta

from app.models.database import get_session, AdminUser
from app.config import API_TOKEN_SECRET

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.get("/token")
async def get_session_token():
    """为对话页生成临时 Session Token"""
    payload = {
        "session_id": str(uuid.uuid4()),
        "type": "session",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, session: Session = Depends(get_session)):
    """管理员/客服登录"""
    stmt = select(AdminUser).where(AdminUser.username == req.username)
    user = session.exec(stmt).first()

    if not user or not bcrypt.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    payload = {
        "sub": user.username,
        "role": user.role,
        "type": "admin",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return TokenResponse(access_token=token, role=user.role)
```

- [ ] **Step 4: 提交**

```bash
git add app/api/ app/dependencies.py
git commit -m "feat: add auth API - session token and JWT login"
```

---

### Task 7: 业务服务层

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/chat_service.py`
- Create: `app/services/stats_service.py`
- Create: `app/__init__.py`

- [ ] **Step 1: 创建 app/services/__init__.py**

```python
```

- [ ] **Step 2: 创建 app/services/chat_service.py**

```python
import time
from datetime import datetime
from sqlmodel import Session, select

from app.agent.graph import run_agent
from app.models.database import Message, Ticket, AuditLog


def process_chat(
    user_message: str,
    session_id: str,
    ip_address: str,
    user_agent: str,
    db: Session,
) -> dict:
    """处理单次对话：执行 Agent + 写入数据库 + 写入日志"""

    # 1. 获取对话历史
    recent_messages = db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(5)
    ).all()
    chat_history = "\n".join(
        [f"{'用户' if m.role == 'user' else '客服'}：{m.content}" for m in reversed(recent_messages)]
    )

    # 2. 执行 Agent
    start_time = time.time()
    result = run_agent(user_message, chat_history)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 3. 保存用户消息
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=user_message,
        intent=result["intent"],
        emotion=result["emotion"],
    )
    db.add(user_msg)
    db.flush()

    # 4. 处理工单创建
    ticket = None
    if result["create_ticket"]:
        ticket = Ticket(
            session_id=session_id,
            user_message=user_message,
            intent=result["intent"],
            emotion=result["emotion"],
            priority=result["ticket_priority"],
            status="pending",
            ai_reply_draft=result["ai_reply"],
        )
        db.add(ticket)
        db.flush()

        # 关联用户消息到工单
        user_msg.ticket_id = ticket.id

    # 5. 保存 AI 回复消息
    assistant_msg = Message(
        session_id=session_id,
        role="assistant",
        content=result["ai_reply"],
        intent=result["intent"],
        emotion=result["emotion"],
        ticket_id=ticket.id if ticket else None,
        response_time_ms=elapsed_ms,
    )
    db.add(assistant_msg)

    # 6. 操作日志
    audit = AuditLog(
        action="ticket_create" if ticket else "chat_submit",
        ip_address=ip_address,
        user_agent=user_agent,
        detail=f"intent={result['intent']}, emotion={result['emotion']}, "
               f"create_ticket={result['create_ticket']}, time_ms={elapsed_ms}",
    )
    db.add(audit)

    db.commit()

    return {
        "message_id": assistant_msg.id,
        "ai_reply": result["ai_reply"],
        "intent": result["intent"],
        "emotion": result["emotion"],
        "create_ticket": result["create_ticket"],
        "ticket_id": ticket.id if ticket else None,
        "ticket_priority": result["ticket_priority"] if ticket else None,
        "response_time_ms": elapsed_ms,
    }
```

- [ ] **Step 3: 创建 app/services/stats_service.py**

```python
from datetime import datetime, timedelta
from sqlmodel import Session, func, select, text

from app.models.database import Message, Ticket


def get_dashboard_stats(db: Session) -> dict:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # 今日对话总数
    total_messages = db.exec(
        select(func.count(Message.id)).where(Message.created_at >= today)
    ).one()

    # 今日工单创建数
    total_tickets = db.exec(
        select(func.count(Ticket.id)).where(Ticket.created_at >= today)
    ).one()

    # AI 直接解决数（今日无工单创建的 AI 回复）
    total_ai_resolved = db.exec(
        select(func.count(Message.id))
        .where(Message.role == "assistant")
        .where(Message.ticket_id == None)
        .where(Message.created_at >= today)
    ).one()

    # AI 解决率
    total_responses = db.exec(
        select(func.count(Message.id))
        .where(Message.role == "assistant")
        .where(Message.created_at >= today)
    ).one()
    ai_resolve_rate = round(total_ai_resolved / total_responses * 100, 1) if total_responses > 0 else 0.0

    # 平均响应时间（毫秒）
    avg_response_time = db.exec(
        select(func.avg(Message.response_time_ms))
        .where(Message.role == "assistant")
        .where(Message.created_at >= today)
    ).one()
    avg_response_time = round(avg_response_time if avg_response_time else 0)

    # 意图分布
    intent_rows = db.exec(
        select(Message.intent, func.count(Message.id))
        .where(Message.role == "user")
        .where(Message.created_at >= today)
        .group_by(Message.intent)
    ).all()
    intent_distribution = [{"name": row[0] or "未知", "count": row[1]} for row in intent_rows]

    # 情绪分布
    emotion_rows = db.exec(
        select(Message.emotion, func.count(Message.id))
        .where(Message.role == "user")
        .where(Message.created_at >= today)
        .group_by(Message.emotion)
    ).all()
    emotion_distribution = [{"name": row[0] or "未知", "count": row[1]} for row in emotion_rows]

    # 高频问题 Top 10（按意图统计）
    top_issues = sorted(intent_distribution, key=lambda x: x["count"], reverse=True)[:10]

    # 最近工单
    recent_tickets = db.exec(
        select(Ticket).order_by(Ticket.created_at.desc()).limit(10)
    ).all()
    recent_tickets_list = [
        {
            "id": t.id,
            "intent": t.intent,
            "priority": t.priority,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
        }
        for t in recent_tickets
    ]

    return {
        "today": {
            "total_messages": total_messages,
            "total_tickets": total_tickets,
            "ai_resolved": total_ai_resolved,
            "ai_resolve_rate": ai_resolve_rate,
            "avg_response_time_ms": avg_response_time,
        },
        "intent_distribution": intent_distribution,
        "emotion_distribution": emotion_distribution,
        "top_issues": top_issues,
        "recent_tickets": recent_tickets_list,
    }
```

- [ ] **Step 4: 提交**

```bash
git add app/services/
git commit -m "feat: add chat service and dashboard stats service"
```

---

### Task 8: API — Chat 对话接口

**Files:**
- Create: `app/api/chat.py`

- [ ] **Step 1: 创建 app/api/chat.py**

```python
from fastapi import APIRouter, Request, Depends
from sqlmodel import Session, select
from pydantic import BaseModel

from app.models.database import get_session, Message, Feedback
from app.services.chat_service import process_chat
from app.dependencies import verify_session_token, rate_limit

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    message_id: int
    ai_reply: str
    intent: str
    emotion: str
    create_ticket: bool
    ticket_id: int | None = None
    ticket_priority: str | None = None
    response_time_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_session_token),
    _rate: None = Depends(rate_limit),
):
    result = process_chat(
        user_message=req.message,
        session_id=req.session_id,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
        db=db,
    )
    return ChatResponse(**result)


class FeedbackRequest(BaseModel):
    rating: str  # good / bad
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    rating: str
    created_at: str


@router.post("/chat/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: int,
    req: FeedbackRequest,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_session_token),
):
    feedback = Feedback(message_id=message_id, rating=req.rating, comment=req.comment)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        created_at=feedback.created_at.isoformat(),
    )


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_session_token),
):
    messages = db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    ).all()
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "emotion": m.emotion,
                "ticket_id": m.ticket_id,
                "response_time_ms": m.response_time_ms,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }
```

- [ ] **Step 2: 注册路由到 main.py**

更新 `app/main.py`，在 `app = FastAPI(...)` 之后、`app.mount(...)` 之前添加：

```python
from app.api import auth, chat, tickets, knowledge, dashboard

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(tickets.router)
app.include_router(knowledge.router)
app.include_router(dashboard.router)
```

- [ ] **Step 3: 提交**

```bash
git add app/api/chat.py app/main.py
git commit -m "feat: add chat API endpoint and register all routers"
```

---

### Task 9: API — Tickets、Knowledge、Dashboard

**Files:**
- Create: `app/api/tickets.py`
- Create: `app/api/knowledge.py`
- Create: `app/api/dashboard.py`

- [ ] **Step 1: 创建 app/api/tickets.py**

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from pydantic import BaseModel
from datetime import datetime

from app.models.database import get_session, Ticket
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


class TicketOut(BaseModel):
    id: int
    session_id: str
    user_message: str
    intent: str
    emotion: str
    priority: str
    status: str
    ai_reply_draft: str
    final_reply: str | None = None
    created_at: str
    resolved_at: str | None = None


class TicketUpdate(BaseModel):
    status: str | None = None
    priority: str | None = None
    final_reply: str | None = None


@router.get("")
async def list_tickets(
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, le=100),
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    stmt = select(Ticket)
    if status:
        stmt = stmt.where(Ticket.status == status)
    stmt = stmt.order_by(Ticket.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    tickets = db.exec(stmt).all()
    total = db.exec(select(Ticket)).all()

    return {
        "items": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "user_message": t.user_message,
                "intent": t.intent,
                "emotion": t.emotion,
                "priority": t.priority,
                "status": t.status,
                "ai_reply_draft": t.ai_reply_draft,
                "final_reply": t.final_reply,
                "created_at": t.created_at.isoformat(),
                "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
            }
            for t in tickets
        ],
        "total": len(total),
        "page": page,
        "page_size": page_size,
    }


@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    return {
        "id": ticket.id,
        "session_id": ticket.session_id,
        "user_message": ticket.user_message,
        "intent": ticket.intent,
        "emotion": ticket.emotion,
        "priority": ticket.priority,
        "status": ticket.status,
        "ai_reply_draft": ticket.ai_reply_draft,
        "final_reply": ticket.final_reply,
        "created_at": ticket.created_at.isoformat(),
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    }


@router.patch("/{ticket_id}")
async def update_ticket(
    ticket_id: int,
    update: TicketUpdate,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")

    if update.status is not None:
        ticket.status = update.status
        if update.status == "resolved":
            ticket.resolved_at = datetime.utcnow()
    if update.priority is not None:
        ticket.priority = update.priority
    if update.final_reply is not None:
        ticket.final_reply = update.final_reply

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return {"ok": True, "ticket_id": ticket.id}
```

- [ ] **Step 2: 创建 app/api/knowledge.py**

```python
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlmodel import Session

from app.models.database import get_session, KnowledgeDocument
from app.rag.loader import load_document
from app.rag.splitter import semantic_split
from app.rag.vectordb import add_documents, delete_collection, get_document_count
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
UPLOAD_DIR = "knowledge_base"


@router.post("/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，仅支持 {ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 10MB")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(filepath, "wb") as f:
        f.write(content)

    # 记录到数据库
    doc = KnowledgeDocument(
        filename=file.filename,
        file_type=ext,
        file_size=len(content),
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        text = load_document(filepath, ext)
        chunks = semantic_split(text, source=file.filename)
        chunk_count = add_documents(chunks)

        doc.chunk_count = chunk_count
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        db.add(doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    db.add(doc)
    db.commit()
    return {"ok": True, "document_id": doc.id, "filename": file.filename, "chunk_count": chunk_count}


@router.get("")
async def list_knowledge(
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    docs = db.exec(
        "SELECT id, filename, file_type, file_size, chunk_count, status, created_at FROM knowledgedocument ORDER BY created_at DESC"
    ).all()
    return {
        "documents": [
            {
                "id": d[0],
                "filename": d[1],
                "file_type": d[2],
                "file_size": d[3],
                "chunk_count": d[4],
                "status": d[5],
                "created_at": d[6],
            }
            for d in docs
        ],
        "total_chunks": get_document_count(),
    }


@router.delete("/{doc_id}")
async def delete_knowledge(
    doc_id: int,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 重建整个知识库（ChromaDB 不支持按 ID 删除单个文档的片段）
    delete_collection()

    # 重新处理剩余文档
    remaining = db.exec(
        f"SELECT id, filename, file_type FROM knowledgedocument WHERE id != {doc_id} AND status = 'ready'"
    ).all()
    for r in remaining:
        filepath = os.path.join(UPLOAD_DIR, r[1])
        if os.path.exists(filepath):
            try:
                text = load_document(filepath, r[2])
                chunks = semantic_split(text, source=r[1])
                add_documents(chunks)
            except Exception:
                pass

    db.delete(doc)
    db.commit()
    return {"ok": True, "deleted_id": doc_id}
```

- [ ] **Step 3: 创建 app/api/dashboard.py**

```python
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.models.database import get_session
from app.services.stats_service import get_dashboard_stats
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/stats")
async def dashboard_stats(
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    return get_dashboard_stats(db)
```

- [ ] **Step 4: 提交**

```bash
git add app/api/tickets.py app/api/knowledge.py app/api/dashboard.py
git commit -m "feat: add tickets, knowledge, dashboard APIs"
```

---

## Sprint 2：前端 + 评测 + 上线

### Task 10: Knowledge Base 文件（Globex 跨境电商）

**Files:**
- Create: `knowledge_base/01-shopping-guide.md`
- Create: `knowledge_base/02-logistics.md`
- Create: `knowledge_base/03-after-sales.md`
- Create: `knowledge_base/04-faq.md`

- [ ] **Step 1: 创建 knowledge_base/01-shopping-guide.md**

```markdown
# Globex 全球购 — 购物指南

## 注册与账户
Globex 支持邮箱和手机号两种注册方式。注册后需完成邮箱验证方可下单。一个账户可绑定多张信用卡和多币种钱包。账户安全问题请前往「账户设置 → 安全中心」查看。

## 下单流程
1. 浏览商品 → 加入购物车
2. 确认订单（商品、数量、颜色/尺码）
3. 填写收货地址（支持海外地址，需用英文或当地语言填写）
4. 选择支付方式（Visa / MasterCard / PayPal / 支付宝 / 微信支付）
5. 确认支付 → 下单成功

## 支付与多币种结算
Globex 支持美元（USD）、欧元（EUR）、人民币（CNY）、日元（JPY）、英镑（GBP）五种货币结算。系统根据用户收货地址自动推荐结算币种。汇率按银行当日牌价计算，下单时锁定汇率，退款时以原支付币种退回，不承担汇率波动损失。

## 订单追踪
下单后可在「我的订单」中实时查看订单状态：
- 待付款：请在 30 分钟内完成支付，超时自动取消
- 配货中：仓库正在拣货打包，通常需要 1-2 个工作日
- 已发货：物流已揽收，可点击查看物流轨迹
- 已签收：包裹已送达，如有问题请在 7 天内联系客服
```

- [ ] **Step 2: 创建 knowledge_base/02-logistics.md**

```markdown
# Globex 全球购 — 物流与清关

## 国际物流时效
| 目的地 | 标准物流 | 快捷物流 |
|--------|---------|---------|
| 北美 | 7-15 个工作日 | 3-5 个工作日 |
| 欧洲 | 7-15 个工作日 | 3-7 个工作日 |
| 东南亚 | 5-10 个工作日 | 2-5 个工作日 |
| 澳洲 | 10-20 个工作日 | 5-10 个工作日 |

以上为预估时效，不含清关时间。实际送达时间可能因天气、海关查验等因素有所延迟。

## 关税与清关
跨境商品可能需要缴纳进口关税和增值税（VAT），具体税率由目的国海关政策决定。关税由收件人承担，Globex 不代收关税。部分国家/地区对一定金额以下的个人自用物品免征关税，具体请查阅当地海关政策。

清关时如遇问题，物流商会通过邮件或电话联系收件人。请在收货后保留清关单据，以备后续退换货需要。

## 包裹追踪
发货后 Gloex 会发送含运单号的确认邮件。可通过以下方式追踪包裹：
- Globex 网站「我的订单」页
- 承运方官网（DHL / FedEx / EMS / 顺丰国际）
- 17TRACK 等第三方查询平台

如超过预计时效 5 个工作日仍未收到包裹，请联系客服启动包裹调查。
```

- [ ] **Step 3: 创建 knowledge_base/03-after-sales.md**

```markdown
# Globex 全球购 — 售后政策

## 退换货规则
- 支持 30 天内无理由退货（自签收之日起计算）
- 商品须为全新未使用状态，包装完整，配件齐全
- 以下商品不支持退货：内衣裤、泳衣、定制商品、已拆封的个人护理产品
- 换货仅支持同款商品不同尺码/颜色的更换

## 跨境退货流程
1. 在「我的订单」中提交退货申请，说明退货原因
2. 客服审核通过后，系统会生成退货标签（Return Label）和退货地址
3. 将商品妥善包装，贴上退货标签，寄送至指定仓库（通常为中国大陆或香港仓库）
4. 仓库签收并检验合格后，7-15 个工作日退款至原支付方式

退货物流费用由用户承担，除非商品存在质量问题或发错货。

## 退款时效
- PayPal / 信用卡：7-15 个工作日
- 支付宝 / 微信支付：3-7 个工作日
- 退款金额为商品实际支付金额，不包含原订单运费

## 争议处理
如对售后处理结果不满意，可通过以下渠道反馈：
- Globex 客服邮箱：support@globex.com
- 在线客服：工作日 9:00-18:00（北京时间）
- 如涉及支付争议，也可联系您的发卡行发起交易争议（Chargeback）
```

- [ ] **Step 4: 创建 knowledge_base/04-faq.md**

```markdown
# Globex 全球购 — 常见问题 FAQ

## Q1：我的包裹显示「清关中」已经 5 天了，正常吗？
清关时间因目的国海关效率而异，一般 1-7 个工作日属正常范围。如超过 7 个工作日，请联系客服并提供运单号，我们将协助查询。

## Q2：收到的商品尺码不合适，怎么换？
请在「我的订单」中提交换货申请，选择需更换的尺码。换货运费由用户承担。建议先确认库存后再寄回商品。

## Q3：退货地址在哪里？
提交退货申请后，系统会自动生成退货标签，上面有完整的退货仓库地址。请勿自行寄送到下单时的发货地址。

## Q4：关税太高了，能退吗？
关税由目的国海关征收，Globex 无法退还关税。建议下单前查阅当地海关政策，了解自用物品免税额度。

## Q5：什么时候发货？
现货商品下单后 1-2 个工作日发货。预售商品以商品页面标注的发货日期为准。

## Q6：支付时汇率怎么算？
按支付时银行当日牌价计算，下单时锁定汇率。最终结算金额以下单页显示的人民币（或其他结算币种）金额为准。

## Q7：可以取消订单吗？
配货中状态前可自行取消订单。如已进入配货中，请联系客服尝试拦截。已发货订单无法取消。

## Q8：包裹显示签收但我没收到？
先确认是否有家人/同事代收。如确认未收到，请在签收记录生成后 48 小时内联系客服，我们将启动调查。

## Q9：快递损坏了怎么办？
请拒收破损包裹并拍照留证，同时联系客服。如已签收后发现损坏，请在 24 小时内拍照并联系客服。

## Q10：多件商品可以合并发货吗？
同一订单内的商品尽量合并发货，但不同仓库的商品可能分开发货。如需合并，请联系客服确认。
```

- [ ] **Step 5: 提交**

```bash
git add knowledge_base/
git commit -m "feat: add Globex cross-border e-commerce knowledge base files"
```

---

### Task 11: 前端 — 登录页

**Files:**
- Create: `frontend/login.html`

- [ ] **Step 1: 创建 frontend/login.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Globex 客服后台 — 登录</title>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .login-card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); width: 400px; max-width: 90vw; }
  .login-card h1 { font-size: 24px; margin-bottom: 8px; color: #1a1a1a; }
  .login-card p { color: #888; margin-bottom: 24px; font-size: 14px; }
  label { display: block; margin-bottom: 4px; font-size: 14px; color: #333; font-weight: 500; }
  input { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }
  input:focus { outline: none; border-color: #1677ff; box-shadow: 0 0 0 2px rgba(22,119,255,0.1); }
  button { width: 100%; padding: 10px; background: #1677ff; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
  button:hover { background: #4096ff; }
  .error { background: #fff2f0; border: 1px solid #ffccc7; color: #ff4d4f; padding: 10px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }
</style>
</head>
<body x-data="{ username: '', password: '', error: '', loading: false }">
<div class="login-card">
  <h1>Globex 客服后台</h1>
  <p>请使用管理员账户登录</p>
  <div x-show="error" class="error" x-text="error"></div>
  <form @submit.prevent="
    loading = true; error = '';
    fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    })
    .then(r => r.ok ? r.json() : r.json().then(j => { throw new Error(j.detail || '登录失败') }))
    .then(data => {
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('role', data.role);
      window.location.href = '/static/admin.html';
    })
    .catch(e => { error = e.message; loading = false; })
  ">
    <label>用户名</label>
    <input type="text" x-model="username" required autocomplete="username">
    <label>密码</label>
    <input type="password" x-model="password" required autocomplete="current-password">
    <button type="submit" x-text="loading ? '登录中...' : '登 录'"></button>
  </form>
</div>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add frontend/login.html
git commit -m "feat: add login page with Alpine.js"
```

---

### Task 12: 前端 — 用户对话页

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: 创建 frontend/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Globex 全球购 — 在线客服</title>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }
  .chat-container { max-width: 720px; margin: 0 auto; height: 100vh; display: flex; flex-direction: column; background: white; }
  .chat-header { padding: 16px 20px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 12px; }
  .chat-header .logo { width: 40px; height: 40px; background: #1677ff; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; }
  .chat-header h2 { font-size: 16px; color: #1a1a1a; }
  .chat-header span { font-size: 12px; color: #52c41a; }
  .chat-messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #1677ff; color: white; border-bottom-right-radius: 4px; }
  .msg.assistant { align-self: flex-start; background: #f0f2f5; color: #333; border-bottom-left-radius: 4px; }
  .msg .meta { font-size: 11px; margin-top: 6px; opacity: 0.7; }
  .ticket-badge { display: inline-block; background: #fff7e6; color: #d48806; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-top: 6px; }
  .chat-input { padding: 16px 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
  .chat-input input { flex: 1; padding: 10px 16px; border: 1px solid #ddd; border-radius: 20px; font-size: 14px; outline: none; }
  .chat-input input:focus { border-color: #1677ff; }
  .chat-input button { padding: 10px 20px; background: #1677ff; color: white; border: none; border-radius: 20px; cursor: pointer; font-size: 14px; }
  .chat-input button:hover { background: #4096ff; }
  .loading-dots { display: flex; gap: 4px; padding: 12px 16px; }
  .loading-dots span { width: 8px; height: 8px; background: #bbb; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
  .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
  .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
</style>
</head>
<body x-data="chatApp()" x-init="init()">
<div class="chat-container">
  <div class="chat-header">
    <div class="logo">G</div>
    <div><h2>Globex 全球购</h2><span>在线客服 · AI 助手</span></div>
  </div>

  <div class="chat-messages" id="messages" x-ref="messages">
    <template x-for="msg in messages" :key="msg.id || msg.tempId">
      <div>
        <div class="msg" :class="msg.role" x-text="msg.content"></div>
        <template x-if="msg.ticket_id">
          <div class="ticket-badge" x-text="'工单 #' + msg.ticket_id"></div>
        </template>
        <template x-if="msg.intent">
          <div class="meta" style="padding-left:16px" x-text="'意图: ' + msg.intent + ' · 情绪: ' + msg.emotion"></div>
        </template>
      </div>
    </template>
    <div x-show="loading" class="msg assistant">
      <div class="loading-dots"><span></span><span></span><span></span></div>
    </div>
  </div>

  <form class="chat-input" @submit.prevent="sendMessage()">
    <input type="text" x-model="input" placeholder="请输入您的问题..." :disabled="loading">
    <button type="submit" :disabled="loading" x-text="loading ? '发送中...' : '发送'"></button>
  </form>
</div>

<script>
function chatApp() {
  return {
    input: '',
    messages: [],
    sessionId: '',
    token: '',
    loading: false,

    async init() {
      // 获取或创建 session_id
      this.sessionId = localStorage.getItem('session_id');
      if (!this.sessionId) {
        this.sessionId = crypto.randomUUID();
        localStorage.setItem('session_id', this.sessionId);
      }
      // 获取 session token
      const resp = await fetch('/api/v1/auth/token');
      const data = await resp.json();
      this.token = data.access_token;
    },

    async sendMessage() {
      if (!this.input.trim() || this.loading) return;
      const content = this.input.trim();
      this.input = '';
      this.loading = true;

      // 添加用户消息
      this.messages.push({ tempId: Date.now(), role: 'user', content });

      try {
        const resp = await fetch('/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this.token },
          body: JSON.stringify({ message: content, session_id: this.sessionId })
        });
        if (!resp.ok) throw new Error('请求失败');
        const data = await resp.json();

        this.messages.push({
          id: data.message_id,
          role: 'assistant',
          content: data.ai_reply,
          intent: data.intent,
          emotion: data.emotion,
          ticket_id: data.ticket_id,
        });
      } catch (e) {
        this.messages.push({ tempId: Date.now(), role: 'assistant', content: '抱歉，系统繁忙，请稍后再试。' });
      }

      this.loading = false;
      this.$nextTick(() => {
        const el = this.$refs.messages;
        el.scrollTop = el.scrollHeight;
      });
    }
  }
}
</script>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add frontend/index.html
git commit -m "feat: add user chat page with HTMX+Alpine.js"
```

---

### Task 13: 前端 — 人工审核台

**Files:**
- Create: `frontend/admin.html`

- [ ] **Step 1: 创建 frontend/admin.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Globex 客服后台 — 工单审核台</title>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }
  .layout { display: flex; height: 100vh; }
  .sidebar { width: 200px; background: #001529; color: white; padding: 20px 0; }
  .sidebar h3 { padding: 0 20px 20px; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1); }
  .sidebar a { display: block; padding: 12px 20px; color: rgba(255,255,255,0.65); text-decoration: none; font-size: 14px; }
  .sidebar a:hover, .sidebar a.active { background: #1677ff; color: white; }
  .main { flex: 1; overflow-y: auto; padding: 24px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
  th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
  th { background: #fafafa; font-weight: 600; }
  .priority-urgent { color: #ff4d4f; font-weight: 600; }
  .priority-high { color: #fa8c16; font-weight: 600; }
  .status-pending { background: #fff7e6; color: #d48806; padding: 2px 8px; border-radius: 4px; }
  .status-resolved { background: #f6ffed; color: #52c41a; padding: 2px 8px; border-radius: 4px; }
  button { padding: 6px 16px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-size: 13px; }
  button.primary { background: #1677ff; color: white; border-color: #1677ff; }
  textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; min-height: 80px; }
</style>
</head>
<body x-data="adminApp()" x-init="checkAuth()">
<div class="layout" x-show="authed">
  <div class="sidebar">
    <h3>Globex 后台</h3>
    <a href="#" class="active">工单审核</a>
    <a href="/static/dashboard.html">数据看板</a>
    <a href="#" @click.prevent="logout()">退出登录</a>
  </div>
  <div class="main">
    <div class="header">
      <h2>工单审核台</h2>
      <select x-model="statusFilter" @change="loadTickets()">
        <option value="">全部状态</option>
        <option value="pending">待处理</option>
        <option value="processing">处理中</option>
        <option value="resolved">已解决</option>
        <option value="closed">已关闭</option>
      </select>
    </div>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>意图</th><th>情绪</th><th>优先级</th><th>状态</th><th>AI 回复草稿</th><th>操作</th>
        </tr>
      </thead>
      <tbody>
        <template x-for="t in tickets" :key="t.id">
          <tr>
            <td x-text="'#' + t.id"></td>
            <td x-text="t.intent"></td>
            <td x-text="t.emotion"></td>
            <td>
              <span :class="'priority-' + t.priority" x-text="t.priority"></span>
            </td>
            <td><span :class="'status-' + t.status" x-text="t.status"></span></td>
            <td style="max-width:300px" x-text="t.ai_reply_draft?.substring(0,80)"></td>
            <td>
              <button @click="openDetail(t)" style="margin-right:4px">详情</button>
              <button class="primary" @click="resolveTicket(t.id)" x-show="t.status !== 'resolved'">解决</button>
            </td>
          </tr>
        </template>
      </tbody>
    </table>

    <!-- 工单详情弹层 -->
    <div x-show="detail" style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center" @click.self="detail=null">
      <div style="background:white;border-radius:12px;padding:24px;width:600px;max-width:90vw;max-height:80vh;overflow-y:auto">
        <h3 x-text="'工单 #'+detail.id"></h3>
        <p><strong>用户消息：</strong><span x-text="detail.user_message"></span></p>
        <p><strong>意图：</strong><span x-text="detail.intent"></span> | <strong>情绪：</strong><span x-text="detail.emotion"></span> | <strong>优先级：</strong><span x-text="detail.priority"></span></p>
        <p><strong>AI 草稿：</strong></p>
        <textarea x-model="detail.ai_reply_draft"></textarea>
        <p style="margin-top:8px"><strong>最终回复：</strong></p>
        <textarea x-model="detail.final_reply"></textarea>
        <div style="margin-top:16px;display:flex;gap:8px">
          <button class="primary" @click="saveDetail()">保存</button>
          <button @click="detail=null">关闭</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
function adminApp() {
  return {
    authed: false,
    tickets: [],
    statusFilter: '',
    detail: null,
    token: '',

    checkAuth() {
      this.token = localStorage.getItem('token');
      if (!this.token) { window.location.href = '/static/login.html'; return; }
      this.authed = true;
      this.loadTickets();
    },

    async loadTickets() {
      let url = '/api/v1/tickets?page=1&page_size=50';
      if (this.statusFilter) url += '&status=' + this.statusFilter;
      const resp = await fetch(url, { headers: { 'Authorization': 'Bearer ' + this.token } });
      const data = await resp.json();
      this.tickets = data.items || [];
    },

    openDetail(t) { this.detail = { ...t }; },

    async saveDetail() {
      await fetch('/api/v1/tickets/' + this.detail.id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this.token },
        body: JSON.stringify({
          ai_reply_draft: this.detail.ai_reply_draft,
          final_reply: this.detail.final_reply,
        })
      });
      this.detail = null;
      this.loadTickets();
    },

    async resolveTicket(id) {
      await fetch('/api/v1/tickets/' + id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this.token },
        body: JSON.stringify({ status: 'resolved' })
      });
      this.loadTickets();
    },

    logout() { localStorage.clear(); window.location.href = '/static/login.html'; }
  }
}
</script>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add frontend/admin.html
git commit -m "feat: add admin ticket review page"
```

---

### Task 14: 前端 — 数据看板

**Files:**
- Create: `frontend/dashboard.html`

- [ ] **Step 1: 创建 frontend/dashboard.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Globex 数据看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }
  .layout { display: flex; height: 100vh; }
  .sidebar { width: 200px; background: #001529; color: white; padding: 20px 0; }
  .sidebar h3 { padding: 0 20px 20px; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1); }
  .sidebar a { display: block; padding: 12px 20px; color: rgba(255,255,255,0.65); text-decoration: none; font-size: 14px; }
  .sidebar a:hover, .sidebar a.active { background: #1677ff; color: white; }
  .main { flex: 1; overflow-y: auto; padding: 24px; }
  .stats-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; }
  .stat-card .value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
  .stat-card .label { font-size: 13px; color: #888; margin-top: 4px; }
  .stat-card .value.green { color: #52c41a; }
  .stat-card .value.blue { color: #1677ff; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .chart-box { background: white; padding: 20px; border-radius: 8px; }
  .chart-box h4 { margin-bottom: 12px; font-size: 14px; color: #333; }
</style>
</head>
<body x-data="dashboardApp()" x-init="checkAuth()">
<div class="layout" x-show="authed">
  <div class="sidebar">
    <h3>Globex 后台</h3>
    <a href="/static/admin.html">工单审核</a>
    <a href="#" class="active">数据看板</a>
    <a href="#" @click.prevent="logout()">退出登录</a>
  </div>
  <div class="main">
    <h2 style="margin-bottom:20px">数据看板</h2>

    <div class="stats-grid">
      <div class="stat-card"><div class="value blue" x-text="stats.today?.total_messages || 0"></div><div class="label">今日对话</div></div>
      <div class="stat-card"><div class="value" x-text="stats.today?.total_tickets || 0"></div><div class="label">今日工单</div></div>
      <div class="stat-card"><div class="value green" x-text="stats.today?.ai_resolved || 0"></div><div class="label">AI 解决</div></div>
      <div class="stat-card"><div class="value green" x-text="(stats.today?.ai_resolve_rate || 0) + '%'"></div><div class="label">解决率</div></div>
      <div class="stat-card"><div class="value blue" x-text="(stats.today?.avg_response_time_ms || 0) + 'ms'"></div><div class="label">平均响应</div></div>
    </div>

    <div class="charts-row">
      <div class="chart-box"><h4>意图分布</h4><canvas id="intentChart"></canvas></div>
      <div class="chart-box"><h4>情绪分布</h4><canvas id="emotionChart"></canvas></div>
    </div>
    <div style="background:white;padding:20px;border-radius:8px">
      <h4 style="margin-bottom:12px">高频问题 Top 10</h4>
      <canvas id="topIssuesChart" style="max-height:250px"></canvas>
    </div>
  </div>
</div>

<script>
function dashboardApp() {
  return {
    authed: false, token: '', stats: {},

    checkAuth() {
      this.token = localStorage.getItem('token');
      if (!this.token) { window.location.href = '/static/login.html'; return; }
      this.authed = true;
      this.loadStats();
    },

    async loadStats() {
      const resp = await fetch('/api/v1/dashboard/stats', { headers: { 'Authorization': 'Bearer ' + this.token } });
      this.stats = await resp.json();
      this.$nextTick(() => this.renderCharts());
    },

    renderCharts() {
      const colors = ['#1677ff','#52c41a','#fa8c16','#ff4d4f','#722ed1','#13c2c2'];

      // 意图分布饼图
      const intentData = this.stats.intent_distribution || [];
      new Chart(document.getElementById('intentChart'), {
        type: 'pie',
        data: { labels: intentData.map(d => d.name), datasets: [{ data: intentData.map(d => d.count), backgroundColor: colors }] },
      });

      // 情绪分布饼图
      const emotionData = this.stats.emotion_distribution || [];
      new Chart(document.getElementById('emotionChart'), {
        type: 'pie',
        data: { labels: emotionData.map(d => d.name), datasets: [{ data: emotionData.map(d => d.count), backgroundColor: ['#52c41a','#faad14','#ff7a45','#ff4d4f'] }] },
      });

      // 高频问题柱状图
      const topData = this.stats.top_issues || [];
      new Chart(document.getElementById('topIssuesChart'), {
        type: 'bar',
        data: { labels: topData.map(d => d.name), datasets: [{ label: '次数', data: topData.map(d => d.count), backgroundColor: '#1677ff' }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false } } }
      });
    },

    logout() { localStorage.clear(); window.location.href = '/static/login.html'; }
  }
}
</script>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add frontend/dashboard.html
git commit -m "feat: add dashboard page with Chart.js"
```

---

### Task 15: Seed Data + 评测脚本

**Files:**
- Create: `seed_data.py`
- Create: `evaluate.py`

- [ ] **Step 1: 创建 seed_data.py（生成管理员 + 导入知识库）**

```python
"""初始化种子数据：创建管理员账户 + 导入知识库文件"""
import os
from passlib.hash import bcrypt
from sqlmodel import Session

from app.models.database import engine, AdminUser, KnowledgeDocument
from app.rag.loader import load_document
from app.rag.splitter import semantic_split
from app.rag.vectordb import add_documents


def create_admin():
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    password_hash = bcrypt.hash(password)

    with Session(engine) as session:
        existing = session.exec(
            f"SELECT id FROM adminuser WHERE username='{username}'"
        ).first()
        if existing:
            print(f"管理员 {username} 已存在，跳过")
            return

        user = AdminUser(username=username, password_hash=password_hash, role="admin")
        session.add(user)
        session.commit()
        print(f"管理员 {username} 已创建 (密码: {password})")


def import_knowledge_base():
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print("knowledge_base/ 目录不存在，跳过")
        return

    with Session(engine) as session:
        for filename in os.listdir(kb_dir):
            if not filename.endswith((".md", ".txt", ".pdf", ".docx")):
                continue

            filepath = os.path.join(kb_dir, filename)
            ext = filename.rsplit(".", 1)[-1]
            file_size = os.path.getsize(filepath)

            existing = session.exec(
                f"SELECT id FROM knowledgedocument WHERE filename='{filename}'"
            ).first()
            if existing:
                print(f"  文件 {filename} 已导入，跳过")
                continue

            doc = KnowledgeDocument(filename=filename, file_type=ext, file_size=file_size, status="processing")
            session.add(doc)
            session.commit()

            try:
                text = load_document(filepath, ext)
                chunks = semantic_split(text, source=filename)
                chunk_count = add_documents(chunks)
                doc.chunk_count = chunk_count
                doc.status = "ready"
            except Exception as e:
                doc.status = "error"
                print(f"  导入 {filename} 失败: {e}")

            session.add(doc)
            session.commit()
            print(f"  已导入 {filename}: {doc.chunk_count} 个片段")


if __name__ == "__main__":
    from app.models.database import init_db
    init_db()

    print("=== 创建管理员 ===")
    create_admin()
    print("\n=== 导入知识库 ===")
    import_knowledge_base()
    print("\n种子数据初始化完成")
```

- [ ] **Step 2: 创建 evaluate.py（50 条测试数据 + 四维评测）**

```python
"""评测脚本：使用 50 条标注数据评测 Agent 性能"""
import json
import time
from app.agent.nodes import detect_intent_and_emotion

# 50 条标注测试数据：[用户消息, 期望意图, 期望情绪, 是否应建工单]
TEST_DATA = [
    # === 咨询类 ===
    ["我的包裹什么时候能到？", "咨询", "平静", False],
    ["这个商品支持支付宝付款吗？", "咨询", "平静", False],
    ["运费是怎么计算的？", "咨询", "平静", False],
    ["欧洲的物流时效大概多久？", "咨询", "平静", False],
    ["下单后多久能发货？", "咨询", "平静", False],
    ["怎么修改收货地址？", "咨询", "平静", False],
    ["支持哪些货币结算？", "咨询", "平静", False],
    ["如果我买两件，能合并发货吗？", "咨询", "平静", False],
    ["美国的关税大概多少？", "咨询", "平静", False],
    ["包裹清关一般要多久？", "咨询", "平静", False],
    ["怎么查订单物流信息？", "咨询", "平静", False],
    ["支付方式有哪些？", "咨询", "平静", False],
    ["注册需要什么信息？", "咨询", "平静", False],

    # === 售后类 ===
    ["尺码不合适，怎么换？", "售后", "平静", False],
    ["退货流程是什么？", "售后", "平静", False],
    ["退款多久能到账？", "售后", "平静", False],
    ["我收到的东西和你描述的不一样", "售后", "不满", False],
    ["30 天无理由退货的规则是什么？", "售后", "平静", False],
    ["退货地址在哪？", "售后", "平静", False],
    ["内衣可以退货吗？", "售后", "平静", False],
    ["退货的运费谁出？", "售后", "平静", False],
    ["退款是按什么汇率退的？", "售后", "平静", False],

    # === 投诉类 ===
    ["包裹丢了！三天了还没到！", "投诉", "愤怒", True],
    ["给我的东西是坏的，你们怎么搞的", "投诉", "不满", True],
    ["关税太高了，你们的价格都是骗人的", "投诉", "不满", False],
    ["客服电话打不通，你们到底有没有人在上班", "投诉", "不满", True],
    ["快递把包裹弄坏了，里面的东西都碎了", "投诉", "愤怒", True],
    ["我在海关那边卡了一周了没人管", "投诉", "不满", True],
    ["为什么别人的运费比我便宜？", "投诉", "不满", False],
    ["你们网站太烂了，支付一直失败", "投诉", "不满", False],

    # === 退款类 ===
    ["我要退款，这个产品我不满意", "退款", "不满", False],
    ["怎么取消订单退款？", "退款", "平静", False],
    ["显示退款成功但我没收到钱", "退款", "不满", True],
    ["我付款了但订单被取消了，钱会退吗？", "退款", "平静", False],
    ["能不能退差价？", "退款", "平静", False],
    ["用 PayPal 退款要多久？", "退款", "平静", False],

    # === 要求转人工 ===
    ["我要找人工客服", "要求转人工", "平静", True],
    ["AI 回答没用，给我转人工", "要求转人工", "不满", True],
    ["帮我转接真人", "要求转人工", "平静", True],
    ["你们的机器人不行，叫我找真人", "要求转人工", "不满", True],

    # === 其他紧急 ===
    ["我账号被盗了，有人用我的卡下了单", "其他", "紧急", True],
    ["我的信用卡被你们多扣了 500 美元", "退款", "紧急", True],
    ["马上帮我取消订单！立刻！", "退款", "紧急", True],
    ["我要报警了，你们是诈骗网站", "投诉", "愤怒", True],
    ["限你们24小时内解决，不然我告你们", "投诉", "愤怒", True],
    ["我隐私泄露了，你们有没有安全保护", "其他", "不满", True],
    ["有安全隐患吗？用户的支付信息安全吗？", "咨询", "平静", False],
    ["春节你们发货吗？", "咨询", "平静", False],
    ["可以送到 PO Box 吗？", "咨询", "平静", False],
    ["怎么联系物流公司？", "咨询", "平静", False],
    ["订单状态为什么两天没更新了？", "咨询", "不满", False],
    ["双十一有什么优惠活动吗？", "咨询", "平静", False],
    ["这个材质是什么？", "咨询", "平静", False],
]


def evaluate():
    print("=" * 60)
    print("AI 客服工单智能处理系统 — 评测报告")
    print("=" * 60)

    # 评测意图 + 情绪
    intent_correct = 0
    emotion_correct = 0
    total = len(TEST_DATA)

    for i, (msg, exp_intent, exp_emotion, _) in enumerate(TEST_DATA):
        result = detect_intent_and_emotion({"user_message": msg})
        actual_intent = result.get("intent", "")
        actual_emotion = result.get("emotion", "")

        if actual_intent == exp_intent:
            intent_correct += 1
        if actual_emotion == exp_emotion:
            emotion_correct += 1

        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{total}] \"{msg[:30]}...\" → 意图:{actual_intent}(期望:{exp_intent}) 情绪:{actual_emotion}(期望:{exp_emotion})")

    intent_acc = round(intent_correct / total * 100, 1)
    emotion_acc = round(emotion_correct / total * 100, 1)

    # 评测 AI 解决率（should_not_create_ticket 视为 AI 可直接解决）
    # 模拟判断：紧急/愤怒 → 应建单，其余看是否匹配知识库
    ticket_expected = sum(1 for _, _, _, need_ticket in TEST_DATA if need_ticket)
    should_not_create = total - ticket_expected

    print("\n" + "=" * 60)
    print("评测结果：")
    print(f"  意图识别准确率: {intent_acc}%  ({intent_correct}/{total})")
    print(f"  情绪识别准确率: {emotion_acc}%  ({emotion_correct}/{total})")
    print(f"  理论 AI 解决率: {round(should_not_create / total * 100, 1)}%  ({should_not_create}/{total})")
    print(f"  回复质量评分: 需通过 LLM-as-Judge 评估（待后续补充）")
    print("=" * 60)

    return {
        "intent_accuracy": intent_acc,
        "emotion_accuracy": emotion_acc,
        "ai_resolve_rate": round(should_not_create / total * 100, 1),
        "total_samples": total,
    }


if __name__ == "__main__":
    evaluate()
```

- [ ] **Step 3: 运行评测脚本验证**

```bash
python evaluate.py
```

- [ ] **Step 4: 提交**

```bash
git add seed_data.py evaluate.py
git commit -m "feat: add seed data script and evaluation script with 50 test cases"
```

---

### Task 16: Docker + README

**Files:**
- Create: `Dockerfile`
- Create: `README.md`

- [ ] **Step 1: 创建 Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data knowledge_base

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 创建 README.md**

```markdown
# AI 客服工单智能处理系统

基于 FastAPI + LangGraph + RAG + DeepSeek 的智能客服系统，面向跨境电商场景，实现对话优先的自动回复与工单兜底路由。

## 快速开始

### 本地开发

```bash
pip install -r requirements.txt
python seed_data.py
uvicorn app.main:app --reload --port 8000
```

访问: http://localhost:8000

- 对话页: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 管理后台: http://localhost:8000/static/login.html (admin / admin123)

### Docker 部署

```bash
docker build -t ai-cs .
docker run -d -p 80:8000 --env-file .env -v ./data:/app/data ai-cs
```

## 环境变量

复制 `.env.example` 为 `.env` 并填入配置：

- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `ADMIN_USERNAME` / `ADMIN_PASSWORD`: 管理后台登录凭据

## 项目结构

```
app/            后端应用
frontend/       前端页面
knowledge_base/ 知识库文件
data/           运行时数据 (SQLite + ChromaDB)
evaluate.py     评测脚本
seed_data.py    种子数据初始化
```

## 技术栈

FastAPI | LangGraph | DeepSeek | ChromaDB | SQLModel | HTMX | Alpine.js | Chart.js

## License

MIT
```

- [ ] **Step 3: 提交**

```bash
git add Dockerfile README.md
git commit -m "feat: add Dockerfile and README"
```

---

## 完成检查清单

在提交最终结果前，验证以下各项：

- [ ] `uvicorn app.main:app --reload` 启动成功
- [ ] `python seed_data.py` 创建管理员 + 导入知识库
- [ ] 浏览器打开 `http://localhost:8000` 对话页正常
- [ ] 发送一条消息，AI 回复正确
- [ ] 打开 `http://localhost:8000/static/login.html` 登录后进入审核台
- [ ] 看板页面图表正常显示
- [ ] `python evaluate.py` 输出四维指标
- [ ] `http://localhost:8000/docs` Swagger 文档可访问

---

*Plan version: 1.0 | 基于设计文档 v2.0 | 2026-05-27*
