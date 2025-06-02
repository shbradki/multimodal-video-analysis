import os
from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[1]
        if parsed.path.startswith("/v/"):
            return parsed.path.split("/v/")[1]
    return None

def get_data_dir() -> str:
    base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base, "backend", "videos")

def get_video_dir(video_id: str) -> str:
    return os.path.join(get_data_dir(), video_id)

def get_frame_dir(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "frames")

def get_transcript_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "transcript.txt")

def get_audio_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "audio.mp3")

def get_summary_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "section_summary.json")

def get_video_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), f"{video_id}.mp4")

def get_frame_embeddings_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "frame_embeddings.json")

def get_transcript_embeddings_path(video_id: str) -> str:
    return os.path.join(get_video_dir(video_id), "transcript_embeddings.json")
