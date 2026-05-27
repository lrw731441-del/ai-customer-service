import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel import Session as SqlSession
import bcrypt
from jose import jwt
from datetime import datetime, timedelta

from app.models.database import get_session, AdminUser, Customer, Session_ as DBSession
from app.config import API_TOKEN_SECRET

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.get("/token")
async def get_session_token():
    """为对话页生成临时 Session Token"""
    payload = {
        "session_id": str(uuid.uuid4()),
        "type": "session",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, session: SqlSession = Depends(get_session)):
    """管理员/客服登录"""
    stmt = select(AdminUser).where(AdminUser.username == req.username)
    user = session.exec(stmt).first()

    if not user or not bcrypt.checkpw(req.password.encode(), user.password_hash.encode()):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    payload = {
        "sub": user.username,
        "role": user.role,
        "type": "admin",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return TokenResponse(access_token=token, role=user.role)


class CustomerAuthRequest(BaseModel):
    name: str
    phone: str


class CustomerAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    customer_id: int
    customer_name: str
    session_id: str


@router.post("/customer", response_model=CustomerAuthResponse)
async def customer_auth(req: CustomerAuthRequest, session: SqlSession = Depends(get_session)):
    """客户认证：姓名 + 手机号/订单号，返回或创建 Customer + Session + JWT"""
    stmt = select(Customer).where(Customer.phone == req.phone)
    customer = session.exec(stmt).first()
    if not customer:
        customer = Customer(name=req.name, phone=req.phone)
        session.add(customer)
        session.commit()
        session.refresh(customer)
    elif customer.name != req.name:
        customer.name = req.name
        session.add(customer)
        session.commit()

    new_session_id = str(uuid.uuid4())
    db_session = DBSession(id=new_session_id, customer_id=customer.id, title="新对话", status="active")
    session.add(db_session)
    session.commit()

    payload = {
        "customer_id": customer.id,
        "name": customer.name,
        "session_id": new_session_id,
        "type": "customer",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return CustomerAuthResponse(
        access_token=token,
        customer_id=customer.id,
        customer_name=customer.name,
        session_id=new_session_id,
    )
