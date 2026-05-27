# AI 客服工单智能处理系统 — 技术设计文档

> 作者：林儒玮 | 日期：2026-05-27 | 版本：v2.0

---

## 一、项目定位

面试项目，目标岗位：AI 应用开发 + 全栈开发。

- 独立开发完成全链路
- 公网可访问，可直接演示
- 企业级质感，但架构精简、不烂尾

---

## 二、核心业务流程：对话优先 + 工单兜底

```
用户发送消息
    │
    ▼
意图识别 + 情绪检测（一次 LLM 调用）
    │
    ├── 情绪=愤怒/紧急 → 跳过 RAG，直接建工单，优先级=紧急，回复模板安抚
    │
    └── 正常情绪 →
            │
            ▼
        RAG 检索知识库
            │
            ├── 命中（相似度 ≥ 0.7）→ AI 生成回复，直接返回，不建工单
            │
            ├── 未命中 → 自动建工单，回复："已为您创建工单，客服稍后处理"
            │
            └── 用户追问"转人工" → 建工单
```

设计原则：**AI 先顶上，能解决不建单。只有搞不定才转人工工单。**

---

## 三、技术选型

| 模块 | 选型 | 理由 |
|------|------|------|
| 后端框架 | FastAPI | 异步支持好，自带 Swagger，企业级 API 标准 |
| Agent 编排 | LangGraph | 有状态多步骤 Agent，面试差异化亮点 |
| LLM | DeepSeek API | 性价比高，中文强，已有 Key |
| 向量数据库 | ChromaDB（嵌入模式） | pip install 即可，零配置，和 FastAPI 同进程 |
| 关系数据库 | SQLite + SQLModel | 开发零配置，ORM 写法与 PostgreSQL 一致，换连接串即切换 |
| 前端 | HTMX + Alpine.js + Chart.js | 零构建工具，CDN 引入，交互现代 + 图表丰富 |
| 消息队列 | 不做 | FastAPI BackgroundTasks 足够 demo 场景 |
| 部署 | Docker 单容器 | 腾讯云轻量 2C2G，已就绪 |

---

## 四、系统架构

### 4.1 架构图

```
用户浏览器（HTMX + Alpine.js）
        │
        ▼
┌──────────────────────────┐
│   FastAPI 服务（单进程）    │
│                          │
│  /api/v1/chat           │ ← 对话接口（核心）
│  /api/v1/tickets        │ ← 工单 CRUD
│  /api/v1/knowledge      │ ← 知识库管理
│  /api/v1/dashboard      │ ← 数据看板
│                          │
│  ┌────────────────────┐  │
│  │  LangGraph Agent    │  │
│  │                    │  │
│  │  Node 1: 意图+情绪 │  │
│  │     │              │  │
│  │     ├─ 正常情绪 ─→ Node 2: RAG 检索  │
│  │     │              │  │
│  │     ├─ 愤怒/紧急/转人工 ─→ 跳过RAG   │
│  │     │              │  │
│  │     └──────────────→ Node 3: 回复+路由 │
│  └────────────────────┘  │
│                          │
│  ┌────────┐ ┌──────────┐ │
│  │ SQLite │ │ ChromaDB │ │
│  │(工单)  │ │(嵌入模式) │ │
│  └────────┘ └──────────┘ │
└──────────────────────────┘
        │
        ▼
    DeepSeek API（外部）
```

### 4.2 数据流

1. 用户打开对话页，前端自动获取 Session Token（无需登录）
2. 用户发送消息到 `POST /api/v1/chat`，携带 Session Token
3. FastAPI 校验 Token + IP 限流检查
4. 调用 LangGraph Agent 执行 3 节点流水线（含条件分支）
5. Agent 返回：AI 直接回复 or 创建工单 + 引导回复
6. 结果存入 SQLite（对话消息 + 工单 + 操作日志），返回前端展示
7. 用户可对 AI 回复提交反馈，用于评测统计

---

## 五、Agent 节点设计

### 5.1 State 定义

```python
class AgentState(TypedDict):
    user_message: str          # 用户原始消息
    intent: str                # 咨询/投诉/退款/售后/要求转人工/其他
    emotion: str               # 平静/不满/愤怒/紧急
    retrieved_docs: list       # RAG 检索到的文档片段
    ai_reply: str              # AI 生成的回复
    create_ticket: bool        # 是否创建工单
    ticket_priority: str       # low/medium/high/urgent
```

### 5.2 三个节点

**Node 1：意图识别 + 情绪检测**

一次 DeepSeek API 调用，结构化输出两个分类：

```
Prompt: 请分析以下用户消息，输出意图和情绪。
意图选项：咨询、投诉、退款、售后、要求转人工、其他
情绪选项：平静、不满、愤怒、紧急

用户消息：{content}

输出 JSON：{"intent": "...", "emotion": "..."}
```

**Node 2：RAG 检索**

- Embedding 模型：DeepSeek API 的 embedding 接口
- 检索策略：纯向量检索（后续可升级混合检索）
- Top-K：取相似度前 5 个文档片段
- 相似度阈值：0.7，低于此值视为未命中

**Node 3：回复生成 + 路由决策**

一次 LLM 调用，同时输出回复内容和路由决策。Node 3 的 Prompt 会将最近 5 条对话历史注入上下文，保证多轮对话的连贯性。

**路由逻辑**：
- 意图=要求转人工 → 直接建工单
- 情绪=愤怒/紧急 → 安抚性回复 + 优先建工单
- 知识库有结果且情绪正常 → 基于检索内容生成专业回复 → 不建工单
- 知识库无结果且情绪正常 → 引导性回复 + 建工单

### 5.3 Agent 流程图（带条件分支）

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("intent_emotion", detect_intent_and_emotion)
workflow.add_node("rag_retrieval", retrieve_knowledge)
workflow.add_node("reply_routing", generate_reply_and_route)

workflow.set_entry_point("intent_emotion")

# 条件边：愤怒/紧急/要求转人工 → 跳过 RAG，直接进入回复+路由
def should_retrieve(state: AgentState) -> str:
    if state["emotion"] in ("愤怒", "紧急") or state["intent"] == "要求转人工":
        return "reply_routing"  # 跳过 RAG
    return "rag_retrieval"

workflow.add_conditional_edges("intent_emotion", should_retrieve, {
    "rag_retrieval": "rag_retrieval",
    "reply_routing": "reply_routing",
})
workflow.add_edge("rag_retrieval", "reply_routing")
workflow.add_edge("reply_routing", END)
```

### 5.4 为什么是 3 个节点不是 5 个

- 意图和情绪是同类任务（分类），合并一次 API 调用，减少 1 次延迟
- 回复和路由天然关联（检索结果直接决定路由），合并一次 API 调用
- 3 次 LLM 调用 → 2 次 LLM 调用，省 33% 延迟和费用
- 面试时：讲清楚为什么这么设计，体现工程判断力

### 5.5 多轮对话与 Session 管理

- 前端进入对话页时，自动获取一个 `session_id`（UUID），存入 localStorage
- 每次发消息带上 session_id，后端按 session 关联对话历史
- Node 3 生成回复时，取同一 session 最近 5 条消息作为上下文注入 Prompt
- 对话页免登录（Session Token 自动获取），管理后台需用户名密码登录（JWT 鉴权）

---

## 六、功能模块

### 6.1 API 列表（12 个端点）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | **核心：发送消息，AI 处理** |
| GET | `/api/v1/tickets` | 工单列表（分页+状态筛选） |
| GET | `/api/v1/tickets/{id}` | 工单详情 |
| PATCH | `/api/v1/tickets/{id}` | 更新工单状态（人工处理） |
| GET | `/api/v1/sessions/{session_id}/messages` | 获取会话完整对话历史 |
| POST | `/api/v1/knowledge/upload` | 上传知识库文档 |
| GET | `/api/v1/knowledge` | 知识库文件列表 |
| DELETE | `/api/v1/knowledge/{id}` | 删除知识库文件 |
| GET | `/api/v1/dashboard/stats` | 看板统计数据 |
| POST | `/api/v1/chat/{message_id}/feedback` | 对 AI 回复提交反馈 |
| POST | `/api/v1/auth/login` | 管理端登录（用户名+密码→JWT） |
| GET | `/api/v1/auth/token` | 获取 Session Token（对话页用） |

### 6.2 知识库管理

**主题**：模拟一家跨境出海电商平台「Globex 全球购」，知识库涵盖跨境电商核心场景。

知识库内容（3-4 个 Markdown 文件，约 3000 字）：
1. **购物指南** — 注册、下单、支付方式、多币种结算、订单追踪
2. **物流与清关** — 国际物流时效、关税说明、清关流程、包裹追踪
3. **售后政策** — 退换货规则、跨境退货流程、退款时效、争议处理
4. **FAQ** — 30 条常见问题（物流延迟、关税纠纷、尺码不符、包裹损坏等）

- 支持格式：PDF / Word / TXT / Markdown
- 文档上传 → 文本解析 → 语义切分 → Embedding → 存入 ChromaDB
- 支持删除和重新上传
- 一个知识库（后续可扩展多租户）

### 6.3 工单管理

- 工单状态：待处理 / 处理中 / 已解决 / 已关闭
- 工单优先级：low / medium / high / urgent
- 每条工单关联对话历史和 AI 回复
- 支持客服手动修改回复后发送

### 6.4 数据看板

- 今日数据：对话总数、工单创建数、AI 直接解决数
- AI 解决率（按日/周/月）
- 平均响应时间趋势图（按小时/天，折线图）
- 高频问题 Top 10（柱状图，基于意图+关键词聚类）
- 各意图类型分布饼图
- 情绪分布（用于预警客户满意度下降）
- 最近工单列表

### 6.5 前端页面（4 个）

| 页面 | 访问者 | 鉴权 | 功能 |
|------|--------|------|------|
| `index.html` | 用户 | 免登录（Session Token） | 对话入口，聊天界面 |
| `login.html` | 客服/管理 | 用户名+密码 | 登录后跳转后台 |
| `admin.html` | 客服 | JWT | 工单审核台，修改回复、关闭工单 |
| `dashboard.html` | 管理 | JWT | 数据看板，图表展示 |

---

## 七、安全设计

| 层级 | 措施 |
|------|------|
| 接入层 | Session Token（对话页自动获取，免登录）+ JWT 鉴权（管理端，登录后签发，24h 过期） |
| 网络层 | IP 限流（每 IP 每分钟最多 30 次请求） |
| 数据层 | 文件上传校验（类型白名单 + 大小限制 10MB），管理员密码 bcrypt 哈希存储 |
| 应用层 | SQLModel 参数化查询，防 SQL 注入 |
| 运维层 | 操作日志全量记录（谁在什么时候做了什么），异常行为告警（短时间大量请求、文件上传异常等） |

---

## 八、部署方案

策略：**本地开发跑通 → Docker 打包 → 云服务器上线**

### 8.1 本地开发（默认方式）

```bash
# 一行启动，零前置依赖（除 Python 3.12）
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 访问 http://localhost:8000/docs 即见 API 文档
```

SQLite + ChromaDB 嵌入模式，无需安装任何数据库服务，全部存本地文件。

### 8.2 云服务器部署（项目跑通后）

- 腾讯云轻量应用服务器 Ubuntu 22.04
- Docker 26，2C2G，50GB 系统盘
- 已就绪

**Docker 单容器打包**：

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

启动：`docker run -d -p 80:8000 --env-file .env -v ./data:/app/data ai-cs`

### 8.3 为什么不拆分多个容器

- 项目本身体量不需要微服务
- 单容器简化部署，2G 内存跑得动
- 面试时说"小规模单容器够用，大规模可拆容器 + K8s"即可

### 8.4 环境变量配置（.env.example）

```bash
# DeepSeek API 配置（必填）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 数据库配置（本地开发用 SQLite，无需修改）
DATABASE_URL=sqlite:///./data/tickets.db

# 向量数据库（ChromaDB 本地持久化路径）
CHROMA_PERSIST_DIR=./data/chroma

# 服务配置
HOST=0.0.0.0
PORT=8000

# 安全配置
API_TOKEN_SECRET=change-me-to-a-random-string
RATE_LIMIT_PER_MINUTE=30
```

---

## 九、数据库模型

```python
# 对话消息表
class Message(SQLModel, table=True):
    id: int = Field(primary_key=True)
    session_id: str        # 会话标识
    role: str              # user / assistant
    content: str
    intent: str | None
    emotion: str | None
    ticket_id: int | None  # 关联工单（可为空）
    created_at: datetime

# 工单表
class Ticket(SQLModel, table=True):
    id: int = Field(primary_key=True)
    session_id: str
    user_message: str
    intent: str
    emotion: str
    priority: str           # low / medium / high / urgent
    status: str             # pending / processing / resolved / closed
    ai_reply_draft: str     # AI 生成的回复草稿
    final_reply: str | None # 最终发送给用户的回复
    created_at: datetime
    resolved_at: datetime | None

# 操作日志表
class AuditLog(SQLModel, table=True):
    id: int = Field(primary_key=True)
    action: str             # 操作类型：chat_submit / ticket_create / knowledge_upload / etc
    ip_address: str
    user_agent: str | None
    detail: str | None      # 操作详情
    created_at: datetime

# 反馈表
class Feedback(SQLModel, table=True):
    id: int = Field(primary_key=True)
    message_id: int         # 关联对话消息
    rating: str             # good / bad
    comment: str | None
    created_at: datetime

# 管理员用户表
class AdminUser(SQLModel, table=True):
    id: int = Field(primary_key=True)
    username: str            # 登录用户名
    password_hash: str       # bcrypt 哈希
    role: str                # agent（客服） / admin（管理员）
    created_at: datetime

# 知识库文件表
class KnowledgeDocument(SQLModel, table=True):
    id: int = Field(primary_key=True)
    filename: str
    file_type: str          # pdf / docx / txt / md
    file_size: int          # 字节数
    chunk_count: int        # 切分后的片段数
    status: str             # processing / ready / error
    created_at: datetime
```

---

## 十、评测体系

用 50 条标注数据做自动评测，面试时有数据可讲：

- **意图识别准确率**：目标 ≥ 85%
- **情绪识别准确率**：目标 ≥ 80%
- **AI 解决率**：目标 ≥ 70%
- **回复质量**：LLM-as-Judge（另一个 LLM 评分 1-5 分）

评测脚本在项目根目录 `evaluate.py`，一键跑出四维指标表格。

---

## 十一、项目目录结构

```
ai-customer-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口 + 生命周期管理
│   ├── config.py            # 配置管理（环境变量）
│   ├── dependencies.py      # 依赖注入（Token鉴权、限流）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # /api/v1/chat 对话接口
│   │   ├── tickets.py       # /api/v1/tickets 工单 CRUD
│   │   ├── knowledge.py     # /api/v1/knowledge 知识库
│   │   ├── dashboard.py     # /api/v1/dashboard 看板
│   │   └── auth.py          # /api/v1/auth Token 管理
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph 状态图定义
│   │   ├── nodes.py         # 三个 Agent 节点实现
│   │   └── prompts.py       # Prompt 模板
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectordb.py      # ChromaDB 操作封装
│   │   ├── loader.py        # 文档解析（PDF/Word/TXT/MD）
│   │   └── splitter.py      # 语义切分
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py      # SQLModel 模型 + 数据库初始化
│   └── services/
│       ├── __init__.py
│       ├── chat_service.py  # 对话业务逻辑
│       └── stats_service.py # 统计数据计算
├── frontend/
│   ├── index.html           # 对话页（用户端）
│   ├── login.html           # 登录页（管理端入口）
│   ├── admin.html           # 人工审核台（客服端）
│   └── dashboard.html       # 数据看板（管理端）
├── knowledge_base/          # 上传的知识库文件存放
├── data/                    # SQLite 数据库 + ChromaDB 持久化文件
├── tests/
│   ├── test_chat.py
│   └── test_tickets.py
├── evaluate.py              # 评测脚本
├── seed_data.py             # 50 条标注数据生成脚本
├── Dockerfile
├── docker-compose.yml       # 可选，本地开发也用单容器
├── requirements.txt
└── README.md
```

---

## 十二、开发计划（2 Sprint）

### Sprint 1（本周）：核心跑通

- [ ] FastAPI 项目骨架搭建
- [ ] SQLite 数据库模型 + 初始化
- [ ] LangGraph 3 节点 Agent 跑通（意图+情绪 → RAG → 回复+路由）
- [ ] `POST /api/v1/chat` 对话接口可用
- [ ] 知识库上传 + ChromaDB 检索
- [ ] 工单 CRUD API
- [ ] 核心接口单元测试（chat + tickets + knowledge）

### Sprint 2（下周）：前端 + 评测 + 发布

- [ ] 4 个前端页面（对话页、登录页、审核台、看板）
- [ ] JWT 登录鉴权 + IP 限流 + bcrypt 密码哈希
- [ ] 操作日志记录 + 异常告警
- [ ] 50 条测试数据 + 评测脚本
- [ ] Docker 打包 + 腾讯云上线
- [ ] README + 演示录制

---

## 十三、面试亮点

1. **对话优先架构**：AI 能解决不建单，真正降低人力成本，而非简单的工单分类
2. **LangGraph 有状态 Agent**：3 节点流水线，节点合并优化（5→3），体现工程判断力
3. **企业级工程化**：从需求到上线独立完成，部署在腾讯云，公网可访问
4. **可量化评测**：50 条标注数据 + 四维自动评测，数据驱动迭代
5. **全栈能力**：FastAPI + HTMX 前端 + Docker 部署 + 数据库设计
6. **扩展空间**：面试时主动讲当前限制（SQLite→PG，向量→混合检索，单容器→K8s），体现架构演进意识

---

## 十四、附录

### 14.1 错误处理策略

LLM 调用失败时采用**降级兜底**，不返回 500 给用户：

| 场景 | 处理方式 |
|------|---------|
| DeepSeek API 超时（>10s） | 返回「抱歉，系统繁忙，请稍后重试。如需紧急帮助请拨打客服热线。」 |
| DeepSeek API 返回错误 | 记录错误日志 → 自动建工单，回复兜底文案 |
| Embedding 服务不可用 | RAG 检索跳过，直接根据意图+情绪生成通用回复 + 建工单 |
| 知识库为空 | 跳过 RAG，所有消息均生成通用回复 + 建工单 |
| 数据库写入失败 | 返回 500 + 错误详情，前端展示「系统异常」 |

所有 LLM 调用异常均写入操作日志，标记 `action="api_error"`，方便排查。

### 14.2 前端交互方式

- 对话页：同步请求，用户发消息后等待 3-5 秒获取 AI 回复，HTMX 天然支持 loading 状态
- 看板页：每次打开页面时拉取最新数据，手动刷新即可
- 审核台：标准表单提交 + 列表分页展示
