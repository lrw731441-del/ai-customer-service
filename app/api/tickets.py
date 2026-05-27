from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from pydantic import BaseModel
from datetime import datetime

from app.models.database import get_session, Ticket
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


class TicketUpdate(BaseModel):
    status: str | None = None
    priority: str | None = None
    final_reply: str | None = None


@router.get("")
async def list_tickets(
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, le=100),
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    stmt = select(Ticket)
    if status:
        stmt = stmt.where(Ticket.status == status)
    stmt = stmt.order_by(Ticket.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    tickets = db.exec(stmt).all()
    total = len(db.exec(select(Ticket)).all())

    return {
        "items": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "user_message": t.user_message,
                "intent": t.intent,
                "emotion": t.emotion,
                "priority": t.priority,
                "status": t.status,
                "ai_reply_draft": t.ai_reply_draft,
                "final_reply": t.final_reply,
                "created_at": t.created_at.isoformat(),
                "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
            }
            for t in tickets
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    return {
        "id": ticket.id,
        "session_id": ticket.session_id,
        "user_message": ticket.user_message,
        "intent": ticket.intent,
        "emotion": ticket.emotion,
        "priority": ticket.priority,
        "status": ticket.status,
        "ai_reply_draft": ticket.ai_reply_draft,
        "final_reply": ticket.final_reply,
        "created_at": ticket.created_at.isoformat(),
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    }


@router.patch("/{ticket_id}")
async def update_ticket(
    ticket_id: int,
    update: TicketUpdate,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")

    if update.status is not None:
        ticket.status = update.status
        if update.status == "resolved":
            ticket.resolved_at = datetime.utcnow()
    if update.priority is not None:
        ticket.priority = update.priority
    if update.final_reply is not None:
        ticket.final_reply = update.final_reply

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return {"ok": True, "ticket_id": ticket.id}
