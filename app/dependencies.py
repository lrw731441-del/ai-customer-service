import time
from collections import defaultdict
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.config import API_TOKEN_SECRET, RATE_LIMIT_PER_MINUTE

security = HTTPBearer(auto_error=False)

# 简易内存限流
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def rate_limit(request: Request):
    """IP 限流：每 IP 每分钟最多 N 次请求"""
    now = time.time()
    ip = request.client.host if request.client else "unknown"

    # 清理 1 分钟前的记录
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < 60]

    if len(_rate_limit_store[ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    _rate_limit_store[ip].append(now)


def verify_session_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 Session Token（对话页用）"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少认证 Token")
    try:
        payload = jwt.decode(credentials.credentials, API_TOKEN_SECRET, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 JWT Token（管理端用）"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="请先登录")
    try:
        payload = jwt.decode(credentials.credentials, API_TOKEN_SECRET, algorithms=["HS256"])
        if payload.get("type") != "admin":
            raise HTTPException(status_code=403, detail="权限不足")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def verify_customer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证客户 Token，提取 customer_id 和 session_id"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="请先进行身份认证")
    try:
        payload = jwt.decode(credentials.credentials, API_TOKEN_SECRET, algorithms=["HS256"])
        if payload.get("type") != "customer":
            raise HTTPException(status_code=403, detail="无效的客户凭证")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")
