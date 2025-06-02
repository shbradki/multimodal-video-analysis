import os
from shared.utils import (
    extract_video_id,
    get_video_path,
    get_frame_dir,
    get_frame_embeddings_path,
)
from backend.core.visual_search import (
    sample_frames,
    embed_frames,
    generate_phrases,
    search_visual,
)

from backend.services.video_service import download_video


def ensure_frame_embeddings(url: str):
    video_id = extract_video_id(url)
    video_path = get_video_path(video_id)
    frame_dir = get_frame_dir(video_id)
    embeddings_path = get_frame_embeddings_path(video_id)

    if not os.path.exists(embeddings_path):

        # Ensure video is downloaded first via video_service

        video_status = download_video(url)
        if video_status.get("status") not in ["exists", "downloaded"]:
            raise RuntimeError(f"Failed to download video for embeddings: {video_status.get('error')}")

        sample_frames(video_path, video_id)
        embed_frames(video_id, frame_dir)

    return {"status": "ready", "video_id": video_id}

def search_scene(url: str, prompt: str, top_k: int = 3):
    video_id = extract_video_id(url)
    queries = generate_phrases(prompt, n=5)
    results = search_visual(video_id, queries, top_k=top_k)
    return {
        "video_id": video_id,
        "query": prompt,
        "matches": [
            {
                "timestamp": ts,
                "score": round(score, 3),
                "url": f"https://www.youtube.com/watch?v={video_id}&t={int(ts)}s"
            } for ts, score in results
        ]
    }
