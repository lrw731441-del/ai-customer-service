import time
from sqlmodel import select

from app.agent.graph import run_agent
from app.models.database import Message, Ticket, AuditLog, TicketReply


def process_chat(
    user_message: str,
    session_id: str,
    customer_id: int,
    ip_address: str,
    user_agent: str,
    db,
) -> dict:
    """处理单次对话：执行 Agent + 写入数据库 + 写入日志"""

    # 1. 获取对话历史
    recent_messages = db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(5)
    ).all()
    chat_history = "\n".join(
        [f"{'用户' if m.role == 'user' else '客服'}：{m.content}" for m in reversed(recent_messages)]
    )

    # 2. 执行 Agent
    start_time = time.time()
    result = run_agent(user_message, chat_history)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 3. 保存用户消息
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=user_message,
        intent=result["intent"],
        emotion=result["emotion"],
    )
    db.add(user_msg)
    db.flush()

    # 4. 处理工单创建
    ticket = None
    if result["create_ticket"]:
        # 先查该会话是否已有未关闭的工单，有则复用
        existing = db.exec(
            select(Ticket)
            .where(Ticket.session_id == session_id)
            .where(Ticket.status.in_(["pending", "processing"]))
            .order_by(Ticket.created_at.desc())
            .limit(1)
        ).first()

        if existing:
            ticket = existing
            # 复用工单时，加一条客户追问记录
            customer_reply = TicketReply(ticket_id=ticket.id, sender="customer", content=user_message)
            db.add(customer_reply)
            if ticket.status == "resolved":
                ticket.status = "processing"
                ticket.resolved_at = None
                db.add(ticket)
        else:
            ticket = Ticket(
                session_id=session_id,
                customer_id=customer_id,
                user_message=user_message,
                intent=result["intent"],
                emotion=result["emotion"],
                priority=result["ticket_priority"],
                status="pending",
                ai_reply_draft=result["ai_reply"],
            )
            db.add(ticket)
            db.flush()
            # 加一条 AI 草稿记录
            ai_reply_record = TicketReply(ticket_id=ticket.id, sender="ai", content=result["ai_reply"])
            db.add(ai_reply_record)

        user_msg.ticket_id = ticket.id

    # 更新会话标题（首条消息）
    from app.models.database import Session_ as DBSession
    db_session = db.get(DBSession, session_id)
    if db_session and db_session.title == "新对话":
        db_session.title = user_message[:30]
        db.add(db_session)

    # 5. 保存 AI 回复消息
    assistant_msg = Message(
        session_id=session_id,
        role="assistant",
        content=result["ai_reply"],
        intent=result["intent"],
        emotion=result["emotion"],
        ticket_id=ticket.id if ticket else None,
        response_time_ms=elapsed_ms,
    )
    db.add(assistant_msg)

    # 6. 操作日志
    audit = AuditLog(
        action="ticket_create" if ticket else "chat_submit",
        ip_address=ip_address,
        user_agent=user_agent,
        detail=f"intent={result['intent']}, emotion={result['emotion']}, "
               f"create_ticket={result['create_ticket']}, time_ms={elapsed_ms}",
    )
    db.add(audit)

    db.commit()

    return {
        "message_id": assistant_msg.id,
        "ai_reply": result["ai_reply"],
        "intent": result["intent"],
        "emotion": result["emotion"],
        "create_ticket": result["create_ticket"],
        "ticket_id": ticket.id if ticket else None,
        "ticket_priority": result["ticket_priority"] if ticket else None,
        "response_time_ms": elapsed_ms,
    }
