# AI 驱动的客服工单智能处理系统

> 作者：林儒玮 | 技术栈：FastAPI + LangGraph + RAG + DeepSeek + Docker | 状态：开发中

---

## 一、项目概述

### 1.1 项目背景

中小型企业客服团队普遍面临以下痛点：
- 重复性问题占客服工作量的 60%~70%，人力浪费严重
- 响应速度慢，客户等待时间长
- 工单分类全靠人工，容易出错、难以追踪
- 知识库更新滞后，客服回答不一致

本项目旨在构建一套企业级 AI 客服工单系统，通过 RAG 知识库检索 + LangGraph 多步骤 Agent + 智能工单路由，实现客服流程的自动化处理，降低人力成本，提升响应质量。

### 1.2 项目定位

- **独立开发**，完整覆盖需求分析 → 系统设计 → 开发实现 → 测试验证 → 公网部署全流程
- **企业级工程化**：Docker 容器化、异步队列、安全防护、监控看板
- **可直接演示**：公网可访问，支持实时上传知识库、提交工单、查看处理结果

### 1.3 核心指标（测试数据集）

| 指标 | 数值 |
|------|------|
| 工单自动解决率 | 待测（目标 ≥ 70%） |
| 平均响应时间 | 待测（目标 ≤ 3s） |
| 意图识别准确率 | 待测（目标 ≥ 85%） |
| 情绪识别准确率 | 待测（目标 ≥ 80%） |
| 系统并发支持 | 待测（目标 ≥ 50 QPS） |

> 📌 以上指标在项目上线后用真实测试数据填充

---

## 二、系统架构

### 2.1 整体架构图

```
用户提交工单
      │
      ▼
┌─────────────────────────────────────────┐
│              FastAPI 网关               │
│     Token鉴权 / IP限流 / 请求校验        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         LangGraph Agent 引擎            │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │ 意图识别  │───▶│  知识库RAG检索   │   │
│  └──────────┘    └──────────────────┘   │
│        │                  │             │
│        ▼                  ▼             │
│  ┌──────────┐    ┌──────────────────┐   │
│  │ 情绪检测  │    │  LLM回复生成     │   │
│  └──────────┘    └──────────────────┘   │
│        │                  │             │
│        └──────────┬───────┘             │
│                   ▼                     │
│          ┌────────────────┐             │
│          │  工单路由分级   │             │
│          └────────────────┘             │
└─────────────────────────────────────────┘
              │              │
      ┌───────┘              └───────┐
      ▼                              ▼
┌──────────┐                  ┌──────────────┐
│ 自动回复  │                  │  人工审核台   │
│ 直接发送  │                  │  草稿+修改    │
└──────────┘                  └──────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│           数据看板 Dashboard             │
│  自动解决率 / 响应时长 / 高频问题 Top10  │
└─────────────────────────────────────────┘
```

### 2.2 数据流说明

1. 用户提交工单文本
2. FastAPI 网关做鉴权、限流、参数校验
3. 工单进入 Redis 异步队列
4. LangGraph Agent 按节点顺序处理：
   - **Node 1 意图识别**：判断工单类型（咨询 / 投诉 / 退款 / 售后 / 其他）
   - **Node 2 情绪检测**：识别用户情绪（平静 / 不满 / 愤怒 / 紧急）
   - **Node 3 RAG 检索**：从知识库中召回相关文档段落
   - **Node 4 回复生成**：基于检索结果 + 意图 + 情绪生成专业回复
   - **Node 5 路由决策**：判断自动处理 or 转人工，并标注优先级
5. 结果写入 PostgreSQL，触发 Webhook 回调

---

## 三、技术选型

### 3.1 技术栈一览

| 模块 | 技术选型 | 选择理由 |
|------|---------|---------|
| 后端框架 | FastAPI | 异步支持好，自带 API 文档，性能优于 Flask |
| Agent 框架 | LangGraph | 支持有状态多步骤 Agent，节点可控，便于调试 |
| LLM | DeepSeek API | 性价比高，中文理解能力强，API 稳定 |
| 向量数据库 | ChromaDB | 支持 metadata 过滤，适合多知识库场景 |
| 消息队列 | Redis | 轻量，异步处理工单，防止高峰期阻塞 |
| 关系数据库 | PostgreSQL | 工单数据持久化，支持复杂查询 |
| 容器化 | Docker + Docker Compose | 一键部署，环境隔离 |
| 前端 | HTML / CSS / JS | 轻量，快速迭代，无需前端框架 |
| 云服务器 | 腾讯云 Ubuntu | 已有运维经验，成本可控 |

### 3.2 RAG 检索策略

采用**混合检索**（Hybrid Search）而非单纯向量检索：

```
用户问题
    │
    ├──▶ BM25 关键词检索（召回精确匹配结果）
    │
    └──▶ 向量语义检索（召回语义相近结果）
              │
              ▼
         RRF 融合排序（Reciprocal Rank Fusion）
              │
              ▼
         Reranker 重排序（Cross-Encoder）
              │
              ▼
         Top-K 段落 → 送入 LLM 生成回复
```

**chunk 策略**：采用语义切分（非固定 token 切分），保留段落完整语义，避免关键信息被截断。

经测试（对比实验）：
- 固定 512 token 切分 vs 语义切分：语义切分检索准确率提升约 X%
- 纯向量检索 vs 混合检索：混合检索准确率提升约 X%

### 3.3 LangGraph Agent 节点设计

```python
# 节点状态定义
class TicketState(TypedDict):
    ticket_id: str
    content: str          # 原始工单内容
    intent: str           # 意图分类结果
    emotion: str          # 情绪检测结果
    retrieved_docs: list  # RAG 检索结果
    reply_draft: str      # AI 生成的回复草稿
    priority: str         # 优先级：low / medium / high / urgent
    route: str            # 路由结果：auto / human

# Agent 流程图
workflow = StateGraph(TicketState)
workflow.add_node("intent_detection", detect_intent)
workflow.add_node("emotion_detection", detect_emotion)
workflow.add_node("rag_retrieval", retrieve_knowledge)
workflow.add_node("reply_generation", generate_reply)
workflow.add_node("routing", route_ticket)
workflow.add_edge("intent_detection", "emotion_detection")
workflow.add_edge("emotion_detection", "rag_retrieval")
workflow.add_edge("rag_retrieval", "reply_generation")
workflow.add_edge("reply_generation", "routing")
```

---

## 四、功能模块详解

### 4.1 工单提交与处理

**API 接口**

```
POST /api/v1/tickets
Authorization: Bearer <token>

Request Body:
{
  "user_id": "string",
  "channel": "web | wechat | email",
  "content": "string",
  "attachments": []
}

Response:
{
  "ticket_id": "string",
  "status": "processing",
  "estimated_time": 3
}
```

**处理流程**：
1. 接收工单 → 写入队列
2. Agent 异步处理（约 2-5 秒）
3. 处理完成后 Webhook 回调通知
4. 前端轮询或 WebSocket 推送结果

### 4.2 知识库管理

支持**零代码接入企业自有知识库**：

```
POST /api/v1/knowledge/upload
支持格式：PDF / Word / TXT / Markdown

处理流程：
上传文件 → 解析文本 → 语义切分 → Embedding → 存入 ChromaDB
```

- 支持创建多个独立知识库（多租户）
- 支持增量更新，无需全量重建索引
- 支持按知识库名称路由（不同业务线使用不同知识库）

### 4.3 工单路由分级

| 路由条件 | 处理方式 |
|---------|---------|
| 意图=咨询 且 情绪=平静 且 知识库命中 | 自动回复，直接发送 |
| 意图=投诉 或 情绪=愤怒 | 转人工，优先级=高 |
| 意图=退款 | 转人工，优先级=中 |
| 知识库无命中 | 转人工，优先级=中 |
| 情绪=紧急 | 转人工，优先级=紧急，触发告警 |

### 4.4 人工审核台

- 展示 AI 生成的回复草稿
- 客服一键发送或手动修改后发送
- 支持标注"AI 回答是否正确"，用于后续优化
- 工单历史记录完整可查

### 4.5 数据看板

实时统计以下指标：
- 工单总量 / 今日新增
- AI 自动解决率（按日/周/月）
- 平均响应时间趋势图
- 高频问题 Top 10
- 各意图类型分布饼图
- 情绪分布（用于预警客户满意度下降）

---

## 五、安全设计

参考布料检索系统的四层安全架构，本系统同样构建多层防护：

| 层级 | 措施 |
|------|------|
| 接入层 | 32位 Token 鉴权，每个客户端独立 Token |
| 网络层 | IP 限流（每IP每分钟最多60次请求） |
| 数据层 | 文件上传类型校验，防止恶意文件注入 |
| 应用层 | SQL 参数化查询，防止注入攻击 |
| 运维层 | 操作日志全量记录，异常行为告警 |

---

## 六、部署方案

### 6.1 目录结构

```
ai-customer-service/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── api/                 # 路由层
│   ├── agent/               # LangGraph Agent
│   ├── rag/                 # RAG 检索模块
│   ├── models/              # 数据库模型
│   └── services/            # 业务逻辑层
├── frontend/
│   ├── index.html           # 工单提交页
│   ├── admin.html           # 人工审核台
│   └── dashboard.html       # 数据看板
├── knowledge_base/          # 知识库文件存储
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── CLAUDE.md                # Claude Code 上下文文件
└── README.md
```

### 6.2 Docker Compose 配置

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  chromadb:
    image: chromadb/chroma
    ports:
      - "8001:8000"

volumes:
  postgres_data:
```

### 6.3 部署步骤

```bash
# 1. 克隆项目
git clone https://github.com/lrw731441-del/ai-customer-service

# 2. 配置环境变量
cp .env.example .env
# 填入 DEEPSEEK_API_KEY 等配置

# 3. 一键启动
docker-compose up -d

# 4. 初始化数据库
docker-compose exec api python init_db.py

# 5. 访问
# Web 演示：http://<your-server-ip>
# API 文档：http://<your-server-ip>/docs
```

---

## 七、开发计划

### 第一周：核心功能

- [ ] FastAPI 项目初始化，基础路由搭建
- [ ] LangGraph Agent 四节点流程跑通
- [ ] DeepSeek API 接入，意图识别 + 情绪检测
- [ ] ChromaDB 向量库初始化，基础 RAG 检索

### 第二周：功能完善

- [ ] 知识库文件上传解析（PDF/Word/TXT）
- [ ] 混合检索策略实现（BM25 + 向量）
- [ ] Redis 异步队列接入
- [ ] 前端：工单提交页 + 人工审核台

### 第三周：工程化上线

- [ ] Docker 容器化配置
- [ ] 数据看板前端开发
- [ ] 腾讯云服务器部署，公网上线
- [ ] 100条模拟工单压测，记录核心指标
- [ ] README 完善 + 演示视频录制

---

## 八、面试亮点总结

1. **LangGraph 有状态 Agent**：实现了五节点处理流水线，支持条件分支路由，区别于简单的 LLM 直接调用

2. **混合检索策略**：BM25 + 向量检索 + RRF 融合排序，相比纯向量检索准确率提升约 X%

3. **语义切分 vs 固定切分**：通过对比实验确定最优 chunk 策略，有数据支撑的技术决策

4. **多租户知识库**：支持企业不同业务线独立维护知识库，架构可扩展

5. **完整工程化落地**：从需求到上线独立完成，具备 Docker 部署、安全防护、监控看板

6. **可扩展设计**：预留 Webhook 回调接口，可对接企业微信 / 钉钉 / 飞书等平台

---

## 九、参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)

---

**GitHub 地址**：https://github.com/lrw731441-del/ai-customer-service（开发中）

**公网演示**：部署完成后更新

---

*文档版本：v1.0 | 更新日期：2026-05-27*
