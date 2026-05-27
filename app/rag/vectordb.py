import uuid
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PERSIST_DIR

import os as _os

_embedding_model = None
_MODEL_NAME = "shibing624/text2vec-base-chinese"


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        # Try to resolve from local cache to avoid network calls
        cache_dir = _os.path.expanduser(
            "~/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots"
        )
        model_path = None
        if _os.path.isdir(cache_dir):
            snapshots = sorted(_os.listdir(cache_dir), reverse=True)
            if snapshots:
                model_path = _os.path.join(cache_dir, snapshots[0])
        _embedding_model = SentenceTransformer(
            model_path if model_path else _MODEL_NAME
        )
    return _embedding_model

_chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=chromadb.Settings(anonymized_telemetry=False),
)

COLLECTION_NAME = "globex_knowledge"


def get_collection():
    """获取或创建知识库集合"""
    return _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def embed_text(text: str) -> List[float]:
    """本地向量化（L2 归一化）"""
    import numpy as np
    vec = _get_model().encode(text)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


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
            if similarity >= 0.10:
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
