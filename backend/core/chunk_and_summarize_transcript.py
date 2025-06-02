import os
import sys
import json
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.utils import (
    get_transcript_path,
    get_summary_path,
    extract_video_id,
)

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

def summarize_chunk(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{
            "role": "user",
            "content": (
                f"Summarize the following transcript segment in a short section title:\n\n{text}\n\n"
                "Your response should only be the short section title and no other explanatory text."
            )
        }],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def chunk_transcript(transcript, max_duration=30):
    chunks = []
    current_chunk = []
    current_start = transcript[0]['start']

    for entry in transcript:
        if current_chunk and (entry['start'] - current_start >= max_duration):
            chunks.append({
                'start': current_start,
                'end': current_chunk[-1]['start'],
                'text': ' '.join([e['text'] for e in current_chunk])
            })
            current_chunk = []
            current_start = entry['start']

        current_chunk.append(entry)

    if current_chunk:
        chunks.append({
            'start': current_start,
            'end': current_chunk[-1]['start'],
            'text': ' '.join([e['text'] for e in current_chunk])
        })

    return chunks

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def summarize_transcript(video_id: str):
    transcript_path = get_transcript_path(video_id)
    summary_path = get_summary_path(video_id)

    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    chunks = chunk_transcript(transcript, max_duration=30)

    sections = []
    for chunk in chunks:
        title = summarize_chunk(chunk["text"])
        sections.append({
            "title": title,
            "start": chunk["start"],
            "end": chunk["end"],
            "timestamp": format_timestamp(chunk["start"])
        })
        print(f"[{format_timestamp(chunk['start'])}] {title}")

    with open(summary_path, "w") as f:
        json.dump(sections, f, indent=2)

    print(f"Saved section summary to {summary_path}")
    return sections

if __name__ == "__main__":
    video_url = input("Enter YouTube URL: ")
    video_id = extract_video_id(video_url)
    summarize_transcript(video_id)
