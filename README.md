# AI 客服工单智能处理系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

基于 **LangGraph + RAG + DeepSeek** 构建的智能客服工单处理系统。面向跨境电商场景，通过 3 节点 Agent 流水线自动完成意图识别、情绪分析、知识库检索与工单路由，实现「AI 优先接管 + 人工兜底升级」的客服自动化闭环。

## 工作流程

```
                     ┌─────────────┐
 用户消息 ──────────→│ 意图识别     │
                     │ + 情绪分析   │
                     └──────┬──────┘
                            │
                    ┌───────┴───────┐
                    │   条件路由     │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │ 常规咨询       │               │ 愤怒 / 紧急 / 转人工
            ▼               │               ▼
     ┌──────────┐           │        ┌──────────┐
     │ 查询重写  │           │        │ 安抚回复  │
     │ RAG 检索 │           │        │ 自动建单  │
     └────┬─────┘           │        └──────────┘
          │                 │
          ▼                 │
     ┌──────────┐           │
     │ 回复生成  │           │
     └────┬─────┘           │
          │                 │
     ┌────┴─────┐           │
     │ 知识匹配? │           │
     └────┬─────┘           │
          │                 │
    ┌─────┴─────┐           │
    │ 是  │ 否  │           │
    ▼     ▼                 │
 AI直答  建单转人工           │
```

## 核心特性

**Agent 决策引擎**
- 3 节点 LangGraph 有状态工作流，TypedDict 定义状态 Schema 确保节点间数据正确传递
- 条件分支：愤怒/紧急用户自动跳过检索直建工单，常规咨询走完整 RAG 管线
- 三维路由决策：综合意图（5 类）、情绪（4 级）、知识库匹配度自动判定处理路径

**RAG 检索增强生成**
- 本地化中文向量检索引擎（ChromaDB + text2vec-base-chinese），无需外部 Embedding API
- 查询重写：短模糊消息自动扩展为搜索关键词，提升召回率
- L2 归一化 + 相似度阈值调优，保证余弦相似度计算准确
- 支持 PDF / Word / Markdown / TXT 文档批量导入与自动分割

**性能优化**
- Embedding 模型导出 ONNX 格式，内存占用降低约 60%（~150MB vs ~400MB）
- 延迟加载 + 本地缓存 + HF 镜像三级加载策略
- 异步 API 设计（FastAPI + thread pool），并发安全

**生产级工程配套**
- JWT 多角色认证（管理员 / 客户 Session），IP 限流，操作审计日志
- 50 条手工标注评测集（5 意图 × 4 情绪），自动化评测脚本
- 管理后台（工单审核台 + 数据看板）+ 客户对话端，完整前后端
- Docker Compose 一键部署，健康检查 + 自动重启

## 快速开始

**环境要求：** Python 3.12+ / DeepSeek API Key

```bash
git clone https://github.com/lrw731441-del/ai-customer-service.git
cd ai-customer-service

pip install -r requirements.txt          # 安装依赖
cp .env.example .env                     # 编辑 .env 填入 DEEPSEEK_API_KEY
python seed_data.py                      # 初始化知识库
uvicorn app.main:app --reload --port 8000
```

- 客户对话：[http://localhost:8000](http://localhost:8000)
- API 文档：[http://localhost:8000/docs](http://localhost:8000/docs)
- 管理后台：[http://localhost:8000/static/login.html](http://localhost:8000/static/login.html)

**Docker 部署：**

```bash
docker compose up -d
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| LLM | DeepSeek (deepseek-chat) |
| 向量数据库 | ChromaDB |
| Embedding | sentence-transformers / text2vec-base-chinese (ONNX) |
| 后端 | FastAPI + SQLModel (SQLite) |
| 前端 | Alpine.js + HTMX + Chart.js |
| 部署 | Docker / Docker Compose |

## 项目结构

```
app/
├── agent/              LangGraph Agent
│   ├── graph.py            工作流定义 + 条件路由
│   ├── nodes.py            3 个核心节点实现
│   └── prompts.py          LLM 提示词模板
├── api/                 FastAPI 路由层
│   ├── auth.py             认证（JWT + 客户注册）
│   ├── chat.py             对话 + 反馈 + 会话管理
│   ├── tickets.py          工单 CRUD + 回复
│   ├── knowledge.py        知识库上传 / 删除
│   └── dashboard.py        数据看板
├── models/              SQLModel 数据模型
├── rag/                 RAG 管线
│   ├── loader.py           文档加载（PDF/Word/MD）
│   ├── splitter.py         语义分割
│   └── vectordb.py         ChromaDB + Embedding
├── services/            业务逻辑层
│   ├── chat_service.py     对话处理 + 工单创建
│   └── stats_service.py    统计聚合
├── config.py            全局配置
└── main.py              应用入口
frontend/                前端静态页面
knowledge_base/          知识库源文件
```

## License

MIT
