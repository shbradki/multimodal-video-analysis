from fastapi import APIRouter, Query
from backend.services import transcript_service

router = APIRouter()

@router.get("/check")
def check_transcript(url: str = Query(..., description="YouTube URL")):
    """
    Check if transcript and summary files exist for a given video.
    """
    return transcript_service.ensure_transcript_and_summary(url)
