import os
from shared.utils import extract_video_id, get_video_path, get_video_dir
import yt_dlp

def download_video(video_url):
    video_id = extract_video_id(video_url)
    video_dir = get_video_dir(video_id)
    os.makedirs(video_dir, exist_ok=True)
    output_path = get_video_path(video_id)

    if not os.path.exists(output_path):
        try:
            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "outtmpl": output_path,
                "merge_output_format": "mp4",
                "quiet": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return {"status": "downloaded", "video_id": video_id, "path": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e), "video_id": video_id}
    else:
        print(f"Video already downloaded: {output_path}")
        return {"status": "exists", "video_id": video_id, "path": output_path}
