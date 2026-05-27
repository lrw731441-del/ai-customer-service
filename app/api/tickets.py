from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from pydantic import BaseModel
from datetime import datetime

from app.models.database import get_session, Ticket, Customer, TicketReply
from app.dependencies import verify_jwt_token, verify_customer_token

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


class TicketUpdate(BaseModel):
    status: str | None = None
    priority: str | None = None
    final_reply: str | None = None


class ReplyRequest(BaseModel):
    content: str


class ReplyResponse(BaseModel):
    id: int
    ticket_id: int
    sender: str
    content: str
    created_at: str


# ─── 管理端：工单列表 ───

@router.get("")
async def list_tickets(
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, le=100),
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    stmt = select(Ticket)
    if status:
        stmt = stmt.where(Ticket.status == status)
    stmt = stmt.order_by(Ticket.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    tickets = db.exec(stmt).all()
    total = len(db.exec(select(Ticket)).all())

    customer_ids = list(set(t.customer_id for t in tickets if t.customer_id))
    customers = {}
    if customer_ids:
        for c in db.exec(select(Customer).where(Customer.id.in_(customer_ids))).all():
            customers[c.id] = c

    return {
        "items": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "customer_name": customers[t.customer_id].name if t.customer_id and t.customer_id in customers else None,
                "customer_phone": customers[t.customer_id].phone if t.customer_id and t.customer_id in customers else None,
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


# ─── 客户端：我的工单 ───

@router.get("/mine")
async def list_my_tickets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, le=100),
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    stmt = select(Ticket).where(
        Ticket.customer_id == token["customer_id"]
    ).order_by(Ticket.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    tickets = db.exec(stmt).all()
    total = len(db.exec(
        select(Ticket).where(Ticket.customer_id == token["customer_id"])
    ).all())

    # 批量获取每个工单的最新回复
    latest_reply_map = {}
    ticket_ids = [t.id for t in tickets]
    if ticket_ids:
        all_replies = db.exec(
            select(TicketReply)
            .where(TicketReply.ticket_id.in_(ticket_ids))
            .where(TicketReply.sender.in_(["admin", "ai"]))
            .order_by(TicketReply.created_at.desc())
        ).all()
        for r in all_replies:
            if r.ticket_id not in latest_reply_map:
                latest_reply_map[r.ticket_id] = r.content

    return {
        "items": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "user_message": t.user_message,
                "intent": t.intent,
                "priority": t.priority,
                "status": t.status,
                "ai_reply_draft": t.ai_reply_draft,
                "final_reply": t.final_reply,
                "latest_reply": latest_reply_map.get(t.id),
                "created_at": t.created_at.isoformat(),
                "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
            }
            for t in tickets
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ─── 管理端：工单详情 ───

@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: int,
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")

    customer = db.get(Customer, ticket.customer_id) if ticket.customer_id else None

    return {
        "id": ticket.id,
        "session_id": ticket.session_id,
        "customer_name": customer.name if customer else None,
        "customer_phone": customer.phone if customer else None,
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


# ─── 管理端：修改工单 ───

@router.patch("/{ticket_id}")
async def update_ticket(
    ticket_id: int,
    update: TicketUpdate,
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")

    if update.status is not None:
        ticket.status = update.status
        if update.status in ("resolved", "closed"):
            ticket.resolved_at = datetime.utcnow()
    if update.priority is not None:
        ticket.priority = update.priority
    if update.final_reply is not None:
        ticket.final_reply = update.final_reply

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return {"ok": True, "ticket_id": ticket.id, "status": ticket.status}


# ─── 管理端：发送回复 ───

@router.post("/{ticket_id}/replies", response_model=ReplyResponse)
async def add_admin_reply(
    ticket_id: int,
    req: ReplyRequest,
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")

    reply = TicketReply(ticket_id=ticket_id, sender="admin", content=req.content)
    db.add(reply)

    if ticket.status == "pending":
        ticket.status = "processing"
        db.add(ticket)

    db.commit()
    db.refresh(reply)
    return ReplyResponse(
        id=reply.id, ticket_id=reply.ticket_id, sender=reply.sender,
        content=reply.content, created_at=reply.created_at.isoformat()
    )


# ─── 管理端：获取回复列表 ───

@router.get("/{ticket_id}/replies")
async def get_replies_admin(
    ticket_id: int,
    db=Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    return _get_replies(ticket_id, db)


# ─── 客户端：发送追问 ───

@router.post("/{ticket_id}/customer-reply", response_model=ReplyResponse)
async def add_customer_reply(
    ticket_id: int,
    req: ReplyRequest,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket or ticket.customer_id != token["customer_id"]:
        raise HTTPException(status_code=404, detail="工单不存在")

    reply = TicketReply(ticket_id=ticket_id, sender="customer", content=req.content)
    db.add(reply)

    if ticket.status == "resolved":
        ticket.status = "processing"
        ticket.resolved_at = None
        db.add(ticket)

    db.commit()
    db.refresh(reply)
    return ReplyResponse(
        id=reply.id, ticket_id=reply.ticket_id, sender=reply.sender,
        content=reply.content, created_at=reply.created_at.isoformat()
    )


# ─── 客户端：获取回复列表 ───

@router.get("/{ticket_id}/customer-replies")
async def get_replies_customer(
    ticket_id: int,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket or ticket.customer_id != token["customer_id"]:
        raise HTTPException(status_code=404, detail="工单不存在")
    return _get_replies(ticket_id, db)


# ─── 客户端：确认解决 ───

@router.patch("/{ticket_id}/confirm")
async def confirm_ticket(
    ticket_id: int,
    db=Depends(get_session),
    token: dict = Depends(verify_customer_token),
):
    ticket = db.get(Ticket, ticket_id)
    if not ticket or ticket.customer_id != token["customer_id"]:
        raise HTTPException(status_code=404, detail="工单不存在")

    ticket.status = "closed"
    ticket.resolved_at = datetime.utcnow()
    db.add(ticket)
    db.commit()
    return {"ok": True, "status": "closed"}


# ─── 共享辅助 ───

def _get_replies(ticket_id: int, db):
    replies = db.exec(
        select(TicketReply)
        .where(TicketReply.ticket_id == ticket_id)
        .order_by(TicketReply.created_at.asc())
    ).all()
    return {
        "replies": [
            {
                "id": r.id,
                "sender": r.sender,
                "content": r.content,
                "created_at": r.created_at.isoformat(),
            }
            for r in replies
        ],
    }
