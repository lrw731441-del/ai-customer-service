# RAG 召回率优化方案

## 问题分析

当前实现为纯向量语义检索（text2vec-base-chinese 768d + ChromaDB cosine），两个瓶颈：

| 瓶颈 | 原因 | 表现 |
|---|---|---|
| 短文本衰落 | 用户输入"快递丢了""怎么办"等短句，语义信息稀薄，向量化后与知识库片段余弦距离偏大 | 相似度普遍 <0.15，空召回 |
| 关键词失配 | 纯语义检索对专有名词、订单号、产品SKU不敏感 | "SKU-8888"跟文档里的"SKU-8888 退货说明"匹配不上 |

## 优化方案：查询重写 + 阈值调优

### 1. 查询重写（核心）

在 `retrieve_knowledge` 节点中，检索前增加一步：让 LLM 将用户短句扩展为搜索关键词串。

```
用户输入: "快递丢了"
    ↓ DeepSeek（一次调用，约 30 tokens）
扩展结果: "包裹物流信息异常 快递丢失 订单未收到 物流跟踪无更新 快递件遗失处理"
    ↓ 用扩展结果做向量检索
召回命中率显著提升
```

实现位置：[app/agent/nodes.py](app/agent/nodes.py) 中的 `retrieve_knowledge` 函数。

```python
def rewrite_query(user_message: str) -> str:
    """将模糊短句扩展为搜索关键词，提升向量检索召回率"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": (
                    "把以下用户消息扩展为用于搜索知识库的关键词短语。"
                    "不要加解释，直接输出关键词，用空格分隔：\n"
                    + user_message
                ),
            }],
            temperature=0.1,
            max_tokens=60,
        )
        expanded = resp.choices[0].message.content.strip()
        return expanded
    except Exception:
        return user_message  # 降级：用原查询
```

改造后的检索流程：

```python
def retrieve_knowledge(state):
    from app.rag.vectordb import search_similar

    user_message = state["user_message"]

    # 1. 查询重写
    search_query = rewrite_query(user_message)

    # 2. 用扩展后的查询做检索
    docs = search_similar(search_query, top_k=8)

    return {"retrieved_docs": docs}
```

### 2. 阈值调优

同时降低相似度阈值、增大 Top-K，作为重写的兜底。

```python
# app/rag/vectordb.py
def search_similar(query: str, top_k: int = 8) -> list:  # 原来 5
    ...
    if similarity >= 0.10:  # 原来 0.15
        docs.append(...)
```

### 3. 效果对比

| 场景 | 优化前 | 优化后 |
|---|---|---|
| "快递丢了" | 相似度 ≤0.10，空召回 → 建工单 | 扩展后命中物流知识库 → AI 解答 |
| "怎么退款" | 不稳定，偶尔命中 | 扩展为"退款流程 退款申请 订单退款 退款到账时间"→ 稳定命中 |
| "SKU-8888 有问题" | 关键词失配，空召回 | 重写后保留 SKU + 补充描述 → 命中 |

## 后续演进方向

当前优化是二维的（查询重写 + 阈值），后续可加入：

- **BM25 关键词检索** (`rank-bm25` + jieba 分词)：与向量检索并行，RRF 融合去重，互补语义和关键词两个维度
- **Cross-Encoder Reranker**：对融合后的候选集做二阶段精排，用 `BAAI/bge-reranker-base` 等模型打分

## 技术要点（面试可用）

1. **为什么需要查询重写**：Embedding 模型对短文本的表示能力有限，用户口语输入信息密度低，直接向量化后在高维空间中容易"飘到"不相关区域
2. **为什么不用 LLM 直接回答**：LLM 的知识有时效性和幻觉问题，RAG 的定位是用知识库约束回答边界，查询重写只是帮检索更精准，不替代检索
3. **降级策略**：重写失败时回退到原始查询，不影响系统可用性
