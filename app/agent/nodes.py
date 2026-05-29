import json
import logging
from typing import List, Dict, Any
from openai import OpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from app.agent.prompts import INTENT_EMOTION_PROMPT, REPLY_ROUTING_PROMPT, FALLBACK_REPLY

logger = logging.getLogger(__name__)
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
        logger.exception("意图识别失败")
        return {"intent": "其他", "emotion": "平静"}


def rewrite_query(user_message: str) -> str:
    """将模糊短句扩展为搜索关键词，提升向量检索召回率"""
    if len(user_message) >= 20:
        return user_message
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
        return expanded if expanded else user_message
    except Exception:
        logger.exception("查询重写失败")
        return user_message


def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: RAG 检索（含查询重写）"""
    from app.rag.vectordb import search_similar

    user_message = state["user_message"]
    search_query = rewrite_query(user_message)
    docs = search_similar(search_query, top_k=8)

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

    # 愤怒/紧急/要求转人工 → 直接建单
    if emotion in ("愤怒", "紧急") or intent == "要求转人工":
        priority = "urgent" if emotion == "紧急" else "high"
        reply = "非常抱歉给您带来不便，我已收到您的反馈，正在为您创建优先处理工单，客服专员将尽快与您联系。"

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
            reply = response.choices[0].message.content.strip()
        except Exception:
            logger.exception("愤怒/紧急用户回复生成失败")

        return {
            "ai_reply": reply,
            "create_ticket": True,
            "ticket_priority": priority,
        }

    # 正常流程：有知识库匹配 → AI 直接答，不建单；无匹配 → 建单转人工
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
        reply = response.choices[0].message.content.strip()
    except Exception:
        logger.exception("回复生成失败")
        reply = FALLBACK_REPLY

    # RAG 检索结果决定是否建工单
    has_knowledge = len(retrieved_docs) > 0
    if has_knowledge:
        return {
            "ai_reply": reply,
            "create_ticket": False,
            "ticket_priority": "low",
        }
    else:
        return {
            "ai_reply": reply,
            "create_ticket": True,
            "ticket_priority": "medium",
        }
