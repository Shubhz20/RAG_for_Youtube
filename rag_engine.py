"""
rag_engine.py — Core backend for YouTube Channel RAG

This file handles everything:
  1. Fetching video IDs from a YouTube channel
  2. Downloading transcripts for each video (captions → Whisper fallback)
  3. Chunking transcripts into smaller pieces
  4. Generating embeddings via OpenAI
  5. Storing chunks + embeddings in ChromaDB
  6. Querying the vector DB and generating grounded answers
"""

import os
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI
import chromadb
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Load the local Whisper model once (reused across all transcriptions)
# "base" is ~150 MB — good balance of speed and accuracy.
# Use "tiny" (~75 MB) if you're low on RAM, or "small" (~500 MB) for better quality.
# ---------------------------------------------------------------------------
_whisper_model = None

def get_whisper_model():
    """Load the Whisper model once and cache it in memory."""
    global _whisper_model
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

    Args:
        channel_url: Any YouTube channel URL — handles all formats:
                     https://www.youtube.com/@ChannelName
                     https://www.youtube.com/@ChannelName/videos
                     https://www.youtube.com/c/ChannelName
                     https://www.youtube.com/channel/UC...
        max_videos:  Cap on how many videos to fetch (default 100)

    Returns:
        List of (video_id, title) tuples.
    """
    # Normalize the URL to always point to the /videos tab
    # Without this, yt-dlp returns tabs (Videos, Live, Shorts) instead of actual videos
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url = url + "/videos"

    ydl_opts = {
        "extract_flat": "in_playlist",  # go one level deep into the playlist
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
        # Only include actual videos (11-char IDs), skip sub-playlists
        if video_id and len(video_id) == 11:
            videos.append((video_id, title))

    return videos


# ---------------------------------------------------------------------------
# Step 2 — Fetch transcript for a single video
# ---------------------------------------------------------------------------

def get_transcript_from_captions(video_id):
    """
    Try to get the transcript from YouTube's built-in captions (free, fast).
    Returns the transcript string, or None if captions aren't available.
    """
    try:
        transcript_pieces = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([piece["text"] for piece in transcript_pieces])
        return full_text
    except Exception:
        return None


def get_transcript_from_whisper(video_id):
    """
    Fallback: download the audio with yt-dlp and transcribe it locally
    using faster-whisper (100% free, no API calls).

    Args:
        video_id: YouTube video ID.

    Returns:
        The transcript string, or None if download/transcription fails.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # Create a temp directory for the audio file
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "audio.mp3")

            # Download audio only, convert to mp3
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",  # low quality = smaller file, faster
                }],
                "quiet": True,
                "no_warnings": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if not os.path.exists(audio_path):
                return None

            # Transcribe locally with faster-whisper (FREE)
            model = get_whisper_model()
            segments, _ = model.transcribe(audio_path, beam_size=3)

            # Combine all segments into a single transcript string
            full_text = " ".join([seg.text.strip() for seg in segments])
            return full_text if full_text.strip() else None

    except Exception:
        return None


def get_transcript(video_id):
    """
    Get the transcript for a video, trying two methods (both FREE):
      1. YouTube captions (instant, no compute needed)
      2. Local Whisper transcription (downloads audio, runs model locally)

    Args:
        video_id: YouTube video ID.

    Returns:
        Tuple of (transcript_text, method) or (None, None).
        method is "captions" or "whisper".
    """
    # Method 1: Try YouTube captions first (free and fast)
    text = get_transcript_from_captions(video_id)
    if text:
        return text, "captions"

    # Method 2: Fall back to local Whisper transcription (also free)
    text = get_transcript_from_whisper(video_id)
    if text:
        return text, "whisper"

    return None, None


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

    For each video:
      - First tries YouTube's built-in captions (free)
      - If no captions, downloads audio and transcribes with Whisper

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
    captions_count = 0
    whisper_count = 0

    for idx, (video_id, title) in enumerate(videos):
        # 2. Get transcript (captions first, then local Whisper fallback)
        transcript, method = get_transcript(video_id)

        if not transcript:
            if progress_callback:
                progress_callback(
                    f"[{idx+1}/{len(videos)}] Failed (no audio/too long): {title}"
                )
            continue

        if method == "captions":
            captions_count += 1
        else:
            whisper_count += 1

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
                    "transcript_method": method,
                }],
            )

        indexed_count += 1
        method_label = "captions" if method == "captions" else "whisper"
        if progress_callback:
            progress_callback(
                f"[{idx+1}/{len(videos)}] Indexed ({method_label}): {title}"
            )

    if progress_callback:
        progress_callback(
            f"\nDone! Indexed {indexed_count}/{len(videos)} videos "
            f"({captions_count} from captions, {whisper_count} from Whisper)."
        )

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
