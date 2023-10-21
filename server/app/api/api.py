from fastapi import APIRouter
from .routes import video_router, prediction_router

router = APIRouter()
router.include_router(video_router, tags=["video"], prefix='/video')
router.include_router(prediction_router, tags=["prediction"], prefix='/prediction')