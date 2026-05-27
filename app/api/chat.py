import uuid
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from sqlmodel import select

from app.models.database import get_session, Message, Feedback, Ticket, Session_ as DBSession
from app.services.chat_service import process_chat
from app.dependencies import verify_session_token, verify_customer_token, verify_jwt_token, rate_limit

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message_id: int
    ai_reply: str
    intent: str
    emotion: str
    create_ticket: bool
    ticket_id: int | None = None
    ticket_priority: str | None = None
    response_time_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
    _rate: None = Depends(rate_limit),
):
    result = process_chat(
        user_message=req.message,
        session_id=token["session_id"],
        customer_id=token["customer_id"],
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
        db=db,
    )
    return ChatResponse(**result)


class TicketStatusResponse(BaseModel):
    ticket_id: int
    status: str
    priority: str
    ai_reply_draft: str
    final_reply: str | None = None
    created_at: str
    resolved_at: str | None = None


@router.get("/tickets/{ticket_id}/status", response_model=TicketStatusResponse)
async def get_ticket_status(
    ticket_id: int,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    return TicketStatusResponse(
        ticket_id=ticket.id,
        status=ticket.status,
        priority=ticket.priority,
        ai_reply_draft=ticket.ai_reply_draft,
        final_reply=ticket.final_reply,
        created_at=ticket.created_at.isoformat(),
        resolved_at=ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    )


class FeedbackRequest(BaseModel):
    rating: str
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    rating: str
    created_at: str


@router.post("/chat/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: int,
    req: FeedbackRequest,
    db=Depends(get_session),
    token: dict = Depends(verify_session_token),
):
    feedback = Feedback(message_id=message_id, rating=req.rating, comment=req.comment)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        created_at=feedback.created_at.isoformat(),
    )


@router.get("/sessions")
async def list_my_sessions(
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    """客户查看自己的会话列表"""
    sessions = db.exec(
        select(DBSession)
        .where(DBSession.customer_id == token["customer_id"])
        .order_by(DBSession.created_at.desc())
    ).all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ],
    }


@router.post("/sessions")
async def create_session(
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    """客户创建新会话"""
    new_id = str(uuid.uuid4())
    db_session = DBSession(
        id=new_id,
        customer_id=token["customer_id"],
        title="新对话",
        status="active",
    )
    db.add(db_session)
    db.commit()
    return {"session_id": new_id, "title": "新对话"}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    """客户删除自己的会话"""
    sess = db.get(DBSession, session_id)
    if not sess or sess.customer_id != token["customer_id"]:
        raise HTTPException(status_code=403, detail="无权操作此会话")
    if sess.id == token["session_id"]:
        raise HTTPException(status_code=400, detail="不能删除当前会话")
    db.delete(sess)
    db.commit()
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    """获取会话历史消息（仅限自己的会话）"""
    sess = db.get(DBSession, session_id)
    if not sess or sess.customer_id != token["customer_id"]:
        raise HTTPException(status_code=403, detail="无权访问此会话")

    return _format_session_messages(session_id, db)


@router.get("/admin/sessions/{session_id}/messages")
async def get_session_messages_admin(
    session_id: str,
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    """管理员获取会话历史消息"""
    return _format_session_messages(session_id, db)


def _format_session_messages(session_id: str, db):
    messages = db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    ).all()
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "emotion": m.emotion,
                "ticket_id": m.ticket_id,
                "response_time_ms": m.response_time_ms,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }
