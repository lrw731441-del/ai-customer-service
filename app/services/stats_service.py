from datetime import datetime
from sqlmodel import Session, func, select

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
