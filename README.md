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
