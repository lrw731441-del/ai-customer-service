from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.models.database import get_session
from app.services.stats_service import get_dashboard_stats
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/stats")
async def dashboard_stats(
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    return get_dashboard_stats(db)
