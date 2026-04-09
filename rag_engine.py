"""
rag_engine.py — Core backend for YouTube Channel RAG

This file handles everything:
  1. Fetching video IDs from a YouTube channel
  2. Downloading transcripts for each video
  3. Chunking transcripts into smaller pieces
  4. Generating embeddings via OpenAI
  5. Storing chunks + embeddings in ChromaDB
  6. Querying the vector DB and generating grounded answers
"""

import os
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI
import chromadb

# ---------------------------------------------------------------------------
# Setup clients
# ---------------------------------------------------------------------------

def get_openai_client():
    """
    Return an OpenAI client.
    Checks three places in order:
      1. Streamlit secrets  (for Streamlit Cloud deployment)
      2. Environment variable (for local .env / export)
      3. Raises a clear error if neither is found
    """
    api_key = None

    # 1. Try Streamlit secrets first (works on Streamlit Cloud)
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    # 2. Fall back to environment variable (works locally with .env)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "For local: create a .env file with OPENAI_API_KEY=sk-... "
            "For Streamlit Cloud: add it in App Settings → Secrets."
        )
    return OpenAI(api_key=api_key)


def get_chroma_collection(collection_name="youtube_channel"):
    """
    Return a ChromaDB collection, creating it if it doesn't exist.
    Uses persistent storage so the index survives restarts.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection


# ---------------------------------------------------------------------------
# Step 1 — Get all video IDs from a channel
# ---------------------------------------------------------------------------

def get_channel_video_ids(channel_url, max_videos=100):
    """
    Use yt-dlp to extract video IDs from a YouTube channel URL.

    Args:
        channel_url: Full URL like "https://www.youtube.com/@ChannelName/videos"
        max_videos:  Cap on how many videos to fetch (default 100)

    Returns:
        List of (video_id, title) tuples.
    """
    ydl_opts = {
        "extract_flat": True,      # don't download, just list
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    videos = []
    for entry in info.get("entries", []):
        video_id = entry.get("id")
        title = entry.get("title", "Untitled")
        if video_id:
            videos.append((video_id, title))

    return videos


# ---------------------------------------------------------------------------
# Step 2 — Fetch transcript for a single video
# ---------------------------------------------------------------------------

def get_transcript(video_id):
    """
    Download the auto-generated transcript for a video.

    Returns the full transcript as a single string, or None if unavailable.
    """
    try:
        transcript_pieces = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([piece["text"] for piece in transcript_pieces])
        return full_text
    except Exception:
        # Some videos have no captions — just skip them
        return None


# ---------------------------------------------------------------------------
# Step 3 — Chunk a long text into smaller overlapping pieces
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into word-level chunks with overlap.

    Args:
        text:       The full transcript string.
        chunk_size: Number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far to advance each iteration

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Generate an embedding for a piece of text
# ---------------------------------------------------------------------------

def get_embedding(text, client=None):
    """
    Call OpenAI to embed a text string.

    Uses 'text-embedding-3-small' — fast, cheap, and good enough for RAG.
    """
    if client is None:
        client = get_openai_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step 5 — Index an entire channel (fetch → chunk → embed → store)
# ---------------------------------------------------------------------------

def index_channel(channel_url, max_videos=100, progress_callback=None):
    """
    End-to-end pipeline: fetch videos, get transcripts, embed, store.

    Args:
        channel_url:       YouTube channel URL.
        max_videos:        Max videos to process.
        progress_callback: Optional function(message) called with status updates.

    Returns:
        Number of videos successfully indexed.
    """
    openai_client = get_openai_client()
    collection = get_chroma_collection()

    # 1. Get video list
    if progress_callback:
        progress_callback("Fetching video list from channel...")
    videos = get_channel_video_ids(channel_url, max_videos=max_videos)
    if progress_callback:
        progress_callback(f"Found {len(videos)} videos. Starting transcription...")

    indexed_count = 0

    for idx, (video_id, title) in enumerate(videos):
        # 2. Get transcript
        transcript = get_transcript(video_id)
        if not transcript:
            if progress_callback:
                progress_callback(f"[{idx+1}/{len(videos)}] Skipped (no transcript): {title}")
            continue

        # 3. Chunk the transcript
        chunks = chunk_text(transcript)

        # 4. Embed and store each chunk
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{video_id}_chunk_{chunk_idx}"

            # Skip if this chunk is already in the DB
            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                continue

            embedding = get_embedding(chunk, client=openai_client)

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[{
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://youtube.com/watch?v={video_id}",
                    "chunk_index": chunk_idx,
                }],
            )

        indexed_count += 1
        if progress_callback:
            progress_callback(f"[{idx+1}/{len(videos)}] Indexed: {title}")

    if progress_callback:
        progress_callback(f"Done! Indexed {indexed_count} videos.")

    return indexed_count


# ---------------------------------------------------------------------------
# Step 6 — Query the indexed channel
# ---------------------------------------------------------------------------

def query_channel(question, n_results=5):
    """
    Search the vector DB for relevant chunks, then ask GPT to answer.

    Args:
        question:  The user's natural-language question.
        n_results: How many chunks to retrieve (top-k).

    Returns:
        dict with "answer" (str) and "sources" (list of dicts).
    """
    openai_client = get_openai_client()
    collection = get_chroma_collection()

    # Check that we have data
    if collection.count() == 0:
        return {
            "answer": "No videos indexed yet. Please index a channel first.",
            "sources": [],
        }

    # Embed the question
    q_embedding = get_embedding(question, client=openai_client)

    # Similarity search
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
    )

    # Build context — tag each chunk with its video title so GPT knows the source
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f'[Video: "{meta["title"]}"]\n{doc}')
    context = "\n\n---\n\n".join(context_parts)

    # Collect unique sources
    seen_urls = set()
    sources = []
    for meta in metadatas:
        url = meta["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": meta["title"], "url": url})

    # Ask GPT for a grounded, contextual answer
    prompt = f"""You are a research assistant that answers questions using YouTube
transcript data. Follow these rules strictly:

1. ONLY use information from the transcript context below — do not add outside knowledge.
2. Use the creator's own words and phrasing where possible. Quote short phrases
   directly (in quotation marks) when they capture the idea well.
3. For EVERY claim or point, cite the specific video title it came from in
   parentheses, e.g. (from "Video Title").
4. If the creator explains the topic across multiple videos, synthesize the
   information and note which video each part comes from.
5. If the context does not contain enough information to answer, say so clearly
   and suggest what the user might search for instead.
6. Keep the answer well-structured: use short paragraphs, not bullet points.

Transcript context:
{context}

Question: {question}

Answer:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
