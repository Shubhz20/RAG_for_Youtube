"""
rag_engine.py — Core backend for YouTube Channel RAG

This file handles everything:
  1. Fetching video IDs from a YouTube channel
  2. Downloading transcripts for each video
  3. Chunking transcripts into smaller pieces
  4. Generating embeddings via OpenAI
  5. Storing chunks + embeddings in ChromaDB
  6. Querying the vector DB and generating grounded answers

Transcript strategy (all FREE):
  - Try English captions first
  - Try any language captions + auto-translate to English
  - Try auto-generated captions in any language + translate
  - Fall back to local Whisper (only if faster-whisper is installed)
"""

import os
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI
import chromadb

# ---------------------------------------------------------------------------
# Optional: local Whisper (works locally, may not install on Streamlit Cloud)
# ---------------------------------------------------------------------------
_whisper_available = False
_whisper_model = None

try:
    from faster_whisper import WhisperModel
    _whisper_available = True
except ImportError:
    pass  # Not installed — that's fine, we'll skip it


def get_whisper_model():
    """Load the Whisper model once and cache it in memory."""
    global _whisper_model
    if not _whisper_available:
        return None
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", compute_type="int8", device="cpu")
    return _whisper_model


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
    Handles all URL formats and normalizes to the /videos tab.
    """
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url = url + "/videos"

    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    videos = []
    for entry in info.get("entries", []):
        video_id = entry.get("id")
        title = entry.get("title", "Untitled")
        if video_id and len(video_id) == 11:
            videos.append((video_id, title))

    return videos


# ---------------------------------------------------------------------------
# Step 2 — Fetch transcript for a single video (multi-strategy)
# ---------------------------------------------------------------------------

def get_transcript_from_captions(video_id):
    """
    Try EVERY available caption track on YouTube, in this order:
      1. English manual captions
      2. English auto-generated captions
      3. Any other language — translated to English
      4. Any auto-generated language — translated to English

    This catches 95%+ of videos. Most "no transcript" errors happen because
    the old code only tried English, but many creators have Hindi, Spanish, etc.

    Returns (transcript_text, method_label) or (None, None).
    """
    try:
        # List all available transcript tracks for this video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # --- Strategy 1: Try to find English captions directly ---
        try:
            transcript = transcript_list.find_transcript(["en"])
            pieces = transcript.fetch()
            text = " ".join([p.text for p in pieces])
            if text.strip():
                return text, "captions (en)"
        except Exception:
            pass

        # --- Strategy 2: Try any manually created transcript, translate to English ---
        try:
            for transcript in transcript_list:
                if not transcript.is_generated:
                    translated = transcript.translate("en").fetch()
                    text = " ".join([p.text for p in translated])
                    if text.strip():
                        return text, f"captions ({transcript.language_code}→en)"
        except Exception:
            pass

        # --- Strategy 3: Try any auto-generated transcript, translate to English ---
        try:
            for transcript in transcript_list:
                if transcript.is_generated:
                    # If it's already English, fetch directly
                    if transcript.language_code.startswith("en"):
                        pieces = transcript.fetch()
                        text = " ".join([p.text for p in pieces])
                    else:
                        translated = transcript.translate("en").fetch()
                        text = " ".join([p.text for p in translated])
                    if text.strip():
                        return text, f"auto ({transcript.language_code}→en)"
        except Exception:
            pass

    except Exception:
        pass

    return None, None


def get_transcript_from_whisper(video_id):
    """
    Last resort: download audio and transcribe with local Whisper model.
    Only runs if faster-whisper is installed. 100% free, no API calls.
    """
    if not _whisper_available:
        return None, None

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "audio.mp3")

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }],
                "quiet": True,
                "no_warnings": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if not os.path.exists(audio_path):
                return None, None

            model = get_whisper_model()
            segments, _ = model.transcribe(audio_path, beam_size=3)
            full_text = " ".join([seg.text.strip() for seg in segments])
            return (full_text, "whisper") if full_text.strip() else (None, None)

    except Exception:
        return None, None


def get_transcript(video_id):
    """
    Get the transcript for a video using all available strategies (all FREE):
      1. YouTube captions — English, any language translated, auto-generated
      2. Local Whisper — downloads audio, transcribes locally (if installed)

    Returns:
        Tuple of (transcript_text, method_label) or (None, None).
    """
    # Try all YouTube caption strategies first
    text, method = get_transcript_from_captions(video_id)
    if text:
        return text, method

    # Fall back to local Whisper if available
    text, method = get_transcript_from_whisper(video_id)
    if text:
        return text, method

    return None, None


# ---------------------------------------------------------------------------
# Step 3 — Chunk a long text into smaller overlapping pieces
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into word-level chunks with overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Generate an embedding for a piece of text
# ---------------------------------------------------------------------------

def get_embedding(text, client=None):
    """Embed a text string using OpenAI text-embedding-3-small."""
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
    """
    openai_client = get_openai_client()
    collection = get_chroma_collection()

    if progress_callback:
        progress_callback("Fetching video list from channel...")
    videos = get_channel_video_ids(channel_url, max_videos=max_videos)
    if progress_callback:
        methods_info = "captions + auto-translate"
        if _whisper_available:
            methods_info += " + local Whisper"
        progress_callback(f"Found {len(videos)} videos. Methods: {methods_info}")

    indexed_count = 0
    method_counts = {}

    for idx, (video_id, title) in enumerate(videos):
        transcript, method = get_transcript(video_id)

        if not transcript:
            if progress_callback:
                progress_callback(
                    f"[{idx+1}/{len(videos)}] Skipped (no transcript found): {title}"
                )
            continue

        # Track which method was used
        method_counts[method] = method_counts.get(method, 0) + 1

        chunks = chunk_text(transcript)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{video_id}_chunk_{chunk_idx}"

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
                    "transcript_method": method,
                }],
            )

        indexed_count += 1
        if progress_callback:
            progress_callback(
                f"[{idx+1}/{len(videos)}] Indexed ({method}): {title}"
            )

    if progress_callback:
        summary = ", ".join(f"{count} via {m}" for m, count in method_counts.items())
        progress_callback(
            f"\nDone! Indexed {indexed_count}/{len(videos)} videos ({summary})."
        )

    return indexed_count


# ---------------------------------------------------------------------------
# Step 6 — Query the indexed channel
# ---------------------------------------------------------------------------

def query_channel(question, n_results=5):
    """Search the vector DB for relevant chunks and generate a grounded answer."""
    openai_client = get_openai_client()
    collection = get_chroma_collection()

    if collection.count() == 0:
        return {
            "answer": "No videos indexed yet. Please index a channel first.",
            "sources": [],
        }

    q_embedding = get_embedding(question, client=openai_client)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f'[Video: "{meta["title"]}"]\n{doc}')
    context = "\n\n---\n\n".join(context_parts)

    seen_urls = set()
    sources = []
    for meta in metadatas:
        url = meta["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": meta["title"], "url": url})

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
