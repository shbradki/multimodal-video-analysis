import os
import sys
import json
import cv2
import torch
import clip
import numpy as np
import yt_dlp
from tqdm import tqdm
from PIL import Image
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.utils import (
    extract_video_id,
    get_video_dir,
    get_frame_dir,
    get_frame_embeddings_path,
    get_video_path,
)

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

# === INIT ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def generate_phrases(prompt, n):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{
            "role": "user",
            "content": f"""Give {n} different phrasings of this scene description for visual search: '{prompt}'
            ONLY PROVIDE THE PHRASING AS A COMMA SEPARATED LIST OF STRINGS, NO OTHER EXPLANATORY TEXT OR INFORMATION"""
        }],
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()
    return [phrase.strip() for phrase in raw.split(',')]

def download_video(video_url):
    video_id = extract_video_id(video_url)
    video_dir = get_video_dir(video_id)
    os.makedirs(video_dir, exist_ok=True)
    output_path = get_video_path(video_id)

    if not os.path.exists(output_path):
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": output_path,
            "merge_output_format": "mp4",
            "quiet": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    else:
        print(f"Video already downloaded: {output_path}")

    return output_path, video_id

def sample_frames(video_path, video_id, every_n_seconds=2):
    frame_dir = get_frame_dir(video_id)
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_seconds)
    frame_count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frame_path = os.path.join(frame_dir, f"{int(timestamp)}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        frame_count += 1

    cap.release()
    print(f"Saved {saved} frames to {frame_dir}")
    return frame_dir

def embed_frames(video_id, frame_dir):
    embeddings_path = get_frame_embeddings_path(video_id)

    embeddings = []
    for fname in tqdm(sorted(os.listdir(frame_dir))):
        if not fname.endswith(".jpg"):
            continue
        timestamp = int(fname.split(".")[0])
        img_path = os.path.join(frame_dir, fname)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(image).cpu().numpy()[0]
        embeddings.append({"timestamp": timestamp, "embedding": emb.tolist()})

    with open(embeddings_path, "w") as f:
        json.dump(embeddings, f, indent=2)
    print(f"Saved embeddings to {embeddings_path}")
    return embeddings_path

def search_visual(video_id, queries, top_k=3):
    embeddings_path = get_frame_embeddings_path(video_id)
    with open(embeddings_path, "r") as f:
        data = json.load(f)

    frame_embs = np.array([item["embedding"] for item in data])
    timestamps = [item["timestamp"] for item in data]

    sims_total = np.zeros(len(frame_embs))
    for query in queries:
        with torch.no_grad():
            text_token = clip.tokenize([query]).to(DEVICE)
            text_emb = model.encode_text(text_token).cpu().numpy()[0]
        sims = frame_embs @ text_emb / (np.linalg.norm(frame_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8)
        sims_total += sims

    sims_avg = sims_total / len(queries)
    top_idxs = np.argsort(sims_avg)[::-1][:top_k]

    return [(timestamps[i], sims_avg[i]) for i in top_idxs]

if __name__ == "__main__":
    video_url = input("Enter YouTube URL: ")
    video_path, video_id = download_video(video_url)
    frame_dir = sample_frames(video_path, video_id, every_n_seconds=1)
    embed_frames(video_id, frame_dir)

    while True:
        query = input("\nDescribe a scene (or 'q' to quit): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break

        queries = generate_phrases(query, n=5)
        results = search_visual(video_id, queries)

        print("\nTop matching timestamps:")
        for ts, score in results:
            print(f"- {ts}s (similarity {score:.3f}) → https://www.youtube.com/watch?v={video_id}&t={ts}s")

# https://www.youtube.com/watch?v=22w7z_lT6YM
