# https://www.youtube.com/watch?v=22w7z_lT6YM

import os
import sys
import json
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.utils import extract_video_id, get_transcript_path, get_video_dir


def fetch_and_save_transcript(url: str) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {video_id}")

    try:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except Exception:
            try:
                
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en-US", "en-GB", "en"])
            except Exception as e:
                raise RuntimeError(f"Error fetching transcript: {e}")

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise RuntimeError(f"Error fetching transcript: {e}")

    video_dir = get_video_dir(video_id)
    os.makedirs(video_dir, exist_ok=True)

    transcript_path = get_transcript_path(video_id)
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)

    return {
        "video_id": video_id,
        "transcript_path": transcript_path,
        "transcript": transcript,
    }

