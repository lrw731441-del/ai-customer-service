import json
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
