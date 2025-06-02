import os
import sys
import json
import faiss
import numpy as np
from openai import OpenAI

# Add root to sys.path to access shared.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.utils import (
    extract_video_id,
    get_summary_path,
)

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def load_sections(video_id: str):
    with open(get_summary_path(video_id), "r") as f:
        return json.load(f)

def build_faiss_index(sections):
    dim = len(get_embedding("test"))
    index = faiss.IndexFlatL2(dim)
    embeddings = []
    metadata = []

    for sec in sections:
        content = sec["title"] + ": " + sec.get("text", "")
        emb = get_embedding(content)
        embeddings.append(emb)
        metadata.append(sec)

    index.add(np.array(embeddings))
    return index, metadata

def query_sections(query: str, index, metadata, top_k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    return [metadata[i] for i in indices[0]]

def ask_gpt(question: str, context_sections):
    context = "\n\n".join(
        f"[{sec['timestamp']}] {sec['title']}" for sec in context_sections
    )
    prompt = (
        "You are a helpful assistant answering questions about a YouTube video.\n"
        "Use the following transcript sections to answer the question. Reference timestamps.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def main():
    video_url = input("Enter YouTube URL: ")
    video_id = extract_video_id(video_url)
    sections = load_sections(video_id)

    print("Embedding sections...")
    index, metadata = build_faiss_index(sections)

    while True:
        question = input("\nAsk a question about the video (or 'q' to quit):\n> ")
        if question.lower() in ['q', 'quit', 'exit']:
            break

        top_sections = query_sections(question, index, metadata, top_k=4)
        answer = ask_gpt(question, top_sections)
        print("\nAnswer:\n" + answer)

if __name__ == "__main__":
    main()
