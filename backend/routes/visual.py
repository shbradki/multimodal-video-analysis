# backend/routes/visual.py

from fastapi import APIRouter, Query
from backend.services import visual_service

router = APIRouter()

@router.get("/ensure-embeddings")
def ensure_embeddings(url: str = Query(..., description="YouTube URL")):
    return visual_service.ensure_frame_embeddings(url)

@router.get("/search")
def search_visual_scene(
    url: str = Query(..., description="YouTube URL"),
    prompt: str = Query(..., description="Scene description"),
    top_k: int = 3
):
    return visual_service.search_scene(url, prompt, top_k)
