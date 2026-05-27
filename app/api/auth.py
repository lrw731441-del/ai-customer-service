import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlmodel import Session, select
import bcrypt
from jose import jwt
from datetime import datetime, timedelta

from app.models.database import get_session, AdminUser
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
async def login(req: LoginRequest, session: Session = Depends(get_session)):
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
