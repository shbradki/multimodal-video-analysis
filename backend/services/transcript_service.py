import os
from shared.utils import extract_video_id, get_transcript_path, get_summary_path
from backend.core.generate_transcript import fetch_and_save_transcript
from backend.core.chunk_and_summarize_transcript import summarize_transcript

def ensure_transcript_and_summary(url: str):
    """
    Ensures both the transcript and section summary are generated and saved for a video.

    Args:
        url (str): YouTube URL

    Returns:
        dict: Status and paths
    """
    video_id = extract_video_id(url)
    transcript_path = get_transcript_path(video_id)
    summary_path = get_summary_path(video_id)

    try:
        # Generate transcript if missing
        if not os.path.exists(transcript_path):
            print(f"[transcript_service] Generating transcript for {video_id}")
            fetch_and_save_transcript(url)  # Pass the full URL

        # Generate summary if missing
        if not os.path.exists(summary_path):
            print(f"[transcript_service] Generating section summary for {video_id}")
            summarize_transcript(video_id)

        return {
            "status": "complete",
            "video_id": video_id,
            "transcript_exists": os.path.exists(transcript_path),
            "summary_exists": os.path.exists(summary_path),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "video_id": video_id,
        }
