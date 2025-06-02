import streamlit as st
import os
import json
import requests
import numpy as np
from openai import OpenAI
from numpy.linalg import norm
import sys

# Add root to sys.path to access shared.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.utils import extract_video_id

# === CONFIG ===
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo-1106"
BACKEND_URL = "http://localhost:8000"

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

# === Utility Functions ===
def format_youtube_link(video_id, start_seconds):
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"

def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(vec1, mat2):
    return mat2 @ vec1 / (norm(mat2, axis=1) * norm(vec1) + 1e-8)

def query_sections(user_query, section_data, top_k=3):
    query_emb = get_embedding(user_query)
    sims = cosine_similarity(query_emb, section_data["embeddings"])
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [section_data["sections"][i] for i in top_idxs]

def ask_gpt(question, context_sections):
    context = "\n\n".join(
        f"[{sec['timestamp']}] {sec['title']}" for sec in context_sections
    )
    prompt = (
        "You are a helpful assistant answering questions about a YouTube video.\n"
        "Use the following transcript sections to answer the question. Cite timestamps.\n\n"
        f"{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

@st.cache_data(show_spinner=False)
def load_sections_with_embeddings(video_id):
    path = f"data/{video_id}/summary.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        sections = json.load(f)
    embeddings = [
        get_embedding(sec["title"] + ": " + sec.get("text", ""))
        for sec in sections
    ]
    return {
        "sections": sections,
        "embeddings": np.array(embeddings)
    }

# === Streamlit UI ===

st.set_page_config(page_title="Multimodal Video Chat", layout="centered")
st.title("🎬 Chat With a YouTube Video")
st.markdown("Paste a YouTube video URL, and ask a question about the transcript or search a visual scene!")

video_url = st.text_input("YouTube Video URL:")

if video_url and len(video_url.strip()) > 0:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Could not extract a valid video ID from the URL.")
    else:
        try:
            with st.spinner("Preparing video (download, transcript, embeddings)..."):
                # Step 1: Download video 
                vid_resp = requests.get(f"{BACKEND_URL}/video/download", params={"url": video_url})
                vid_data = vid_resp.json()
                if vid_resp.status_code != 200 or vid_data.get("status") not in ["exists", "downloaded"]:
                    raise RuntimeError(f"Video download failed: {vid_data.get('error', 'Unknown error')}")

                # Step 2: Check transcript (generate if missing)
                tr_resp = requests.get(f"{BACKEND_URL}/transcript/check", params={"url": video_url})
                if tr_resp.status_code != 200:
                    raise RuntimeError("Failed to check/generate transcript.")

                # Step 3: Ensure visual embeddings
                vi_resp = requests.get(f"{BACKEND_URL}/visual/ensure-embeddings", params={"url": video_url})
                if vi_resp.status_code != 200:
                    raise RuntimeError("Failed to ensure frame embeddings.")


            section_data = load_sections_with_embeddings(video_id)

            tab1, tab2 = st.tabs(["🧠 Ask about Transcript", "🖼️ Search Visual Scene"])

            with tab1:
                question = st.text_input("Ask a question about the video:", key="transcript_question")
                if question and section_data:
                    with st.spinner("Thinking..."):
                        top_sections = query_sections(question, section_data)
                        answer = ask_gpt(question, top_sections)

                    st.markdown("### 💬 Answer")
                    st.markdown(answer)

                    st.markdown("### 🔗 Referenced Sections")
                    for sec in top_sections:
                        link = format_youtube_link(video_id, sec["start"])
                        st.markdown(f"- [{sec['timestamp']}] [{sec['title']}]({link})")

            with tab2:
                scene_query = st.text_input("Describe a scene to search:", key="scene_query")
                if scene_query:
                    with st.spinner("Searching visually..."):
                        response = requests.get(
                            f"{BACKEND_URL}/visual/search",
                            params={"url": video_url, "prompt": scene_query}
                        )
                        matches = response.json().get("matches", []) if response.status_code == 200 else []

                    st.markdown("### ⏱️ Matching Scenes")
                    for ts_info in matches:
                        ts = ts_info["timestamp"]
                        score = ts_info["score"]
                        link = format_youtube_link(video_id, ts)
                        st.markdown(f"- [Scene at {int(ts)}s]({link}) (score: {score:.3f})")

        except Exception as e:
            st.error(f"Something went wrong preparing the video: {e}")

else:
    st.info("Please enter a YouTube link to begin.")
