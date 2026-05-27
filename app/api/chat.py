from fastapi import APIRouter, Request, Depends
from sqlmodel import Session, select
from pydantic import BaseModel

from app.models.database import get_session, Message, Feedback
from app.services.chat_service import process_chat
from app.dependencies import verify_session_token, rate_limit

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


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
    db: Session = Depends(get_session),
    token: dict = Depends(verify_session_token),
    _rate: None = Depends(rate_limit),
):
    result = process_chat(
        user_message=req.message,
        session_id=req.session_id,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
        db=db,
    )
    return ChatResponse(**result)


class FeedbackRequest(BaseModel):
    rating: str  # good / bad
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
    db: Session = Depends(get_session),
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


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_session_token),
):
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
