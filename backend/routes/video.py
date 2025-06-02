from fastapi import APIRouter, Query
from backend.services import video_service

router = APIRouter()

@router.get("/download")
def download_video(url: str = Query(..., description="YouTube URL")):
    return video_service.download_video(url)
