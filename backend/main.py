import sys
import os
from fastapi import FastAPI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.routes.video import router as video_router
from backend.routes.transcript import router as transcript_router
from backend.routes.visual import router as visual_router

app = FastAPI()
app.include_router(video_router, prefix="/video")
app.include_router(transcript_router, prefix="/transcript")
app.include_router(visual_router, prefix="/visual")


