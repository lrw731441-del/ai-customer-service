from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END

from app.agent.nodes import detect_intent_and_emotion, retrieve_knowledge, generate_reply_and_route


class AgentState(TypedDict):
    user_message: str
    chat_history: str
    intent: str
    emotion: str
    retrieved_docs: list
    ai_reply: str
    create_ticket: bool
    ticket_priority: str


def should_retrieve(state: AgentState) -> str:
    """条件路由：愤怒/紧急/转人工 → 跳过 RAG"""
    if state.get("emotion") in ("愤怒", "紧急") or state.get("intent") == "要求转人工":
        return "reply_routing"
    return "rag_retrieval"


def build_graph():
    workflow = StateGraph(AgentState)

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
        "intent": "",
        "emotion": "",
        "retrieved_docs": [],
        "ai_reply": "",
        "create_ticket": False,
        "ticket_priority": "low",
    })
    return {
        "intent": result.get("intent", "其他"),
        "emotion": result.get("emotion", "平静"),
        "ai_reply": result.get("ai_reply", ""),
        "create_ticket": result.get("create_ticket", False),
        "ticket_priority": result.get("ticket_priority", "low"),
    }
