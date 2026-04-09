"""
rag_engine.py — Core backend for YouTube Channel RAG

100% FREE stack:
  - Transcripts  : yt-dlp subtitle download (no API key needed)
  - Embeddings   : ChromaDB built-in DefaultEmbeddingFunction
                   (uses all-MiniLM-L6-v2 via onnxruntime — no sentence_transformers needed,
                    works on Python 3.14+, runs locally, free)
  - Vector DB    : ChromaDB (local, free)
  - LLM answers  : Groq API — free tier, just needs a free account at console.groq.com
"""

import os
import re
import tempfile
from yt_dlp import YoutubeDL
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq

# ---------------------------------------------------------------------------
# Load the embedding function once and cache it (free, runs locally)
# ChromaDB's DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via onnxruntime.
# No separate sentence_transformers install needed.
# ---------------------------------------------------------------------------
_embed_fn = None

def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = DefaultEmbeddingFunction()
    return _embed_fn


# ---------------------------------------------------------------------------
# Optional: local Whisper (works locally, skips gracefully on cloud)
# ---------------------------------------------------------------------------
_whisper_available = False
_whisper_model = None

try:
    from faster_whisper import WhisperModel
    _whisper_available = True
except ImportError:
    pass


def get_whisper_model():
    global _whisper_model
    if not _whisper_available:
        return None
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", compute_type="int8", device="cpu")
    return _whisper_model


# ---------------------------------------------------------------------------
# Setup clients
# ---------------------------------------------------------------------------

def get_groq_client():
    """
    Return a Groq client using GROQ_API_KEY.
    Get a free key at: https://console.groq.com
    Add it to Streamlit Cloud Secrets as:  GROQ_API_KEY = "gsk_..."
    """
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Get a free key at https://console.groq.com "
            "then add it to Streamlit Secrets or your .env file."
        )
    return Groq(api_key=api_key)


def get_chroma_collection(collection_name="youtube_channel"):
    """Return (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name=collection_name)


def get_youtube_cookies_path():
    """
    Optional: pass YouTube cookies to yt-dlp to avoid 403 errors on cloud IPs.

    How to get your cookies:
      1. Install browser extension "Get cookies.txt LOCALLY"
      2. Open youtube.com while logged in → export cookies
      3. Paste the full text into Streamlit Secrets as:
            YOUTUBE_COOKIES = \"\"\"
            # Netscape HTTP Cookie File
            .youtube.com  TRUE  /  ...
            \"\"\"
    Returns None if not configured.
    """
    cookies_content = None
    try:
        import streamlit as st
        cookies_content = st.secrets.get("YOUTUBE_COOKIES")
    except Exception:
        pass
    if not cookies_content:
        cookies_content = os.environ.get("YOUTUBE_COOKIES")
    if not cookies_content:
        return None

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(cookies_content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Step 1 — Get all video IDs from a channel
# ---------------------------------------------------------------------------

def get_channel_video_ids(channel_url, max_videos=100):
    """Extract video IDs from any YouTube channel URL format."""
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url = url + "/videos"

    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
    }
    cookies_path = get_youtube_cookies_path()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return [
        (e["id"], e.get("title", "Untitled"))
        for e in info.get("entries", [])
        if e.get("id") and len(e["id"]) == 11
    ]


# ---------------------------------------------------------------------------
# Step 2 — Fetch subtitles via yt-dlp
# ---------------------------------------------------------------------------

def parse_vtt(vtt_text):
    """Convert WebVTT subtitle text to clean plain text."""
    text_lines = []
    for line in vtt_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "NOTE", "STYLE")):
            continue
        if "-->" in line or re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            text_lines.append(line)

    # Remove consecutive duplicate lines (common in auto-captions)
    deduped, prev = [], None
    for line in text_lines:
        if line != prev:
            deduped.append(line)
            prev = line

    return " ".join(deduped)


def get_transcript_from_subtitles(video_id):
    """
    Download subtitles using yt-dlp and return plain text.
    Tries English first, then any available language.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "subtitleslangs": [
                    "en", "en-US", "en-GB", "en-IN",
                    "hi", "es", "fr", "de", "pt", "ja", "ko",
                    "zh-Hans", "zh-Hant", "ar", "ru",
                ],
                "outtmpl": os.path.join(tmp_dir, "sub"),
                "quiet": True,
                "no_warnings": True,
            }
            cookies_path = get_youtube_cookies_path()
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            vtt_files = [f for f in os.listdir(tmp_dir) if f.endswith(".vtt")]
            if not vtt_files:
                return None, None

            en_files = [f for f in vtt_files if ".en" in f.lower()]
            chosen = en_files[0] if en_files else vtt_files[0]
            lang = "en" if en_files else chosen.split(".")[-2]

            with open(os.path.join(tmp_dir, chosen), "r", encoding="utf-8") as f:
                text = parse_vtt(f.read())

            return (text, f"subtitles ({lang})") if text.strip() else (None, None)
    except Exception:
        return None, None


def get_transcript_from_whisper(video_id):
    """Last-resort: local Whisper audio transcription (only if installed)."""
    if not _whisper_available:
        return None, None
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
                "postprocessors": [{"key": "FFmpegExtractAudio",
                                    "preferredcodec": "mp3",
                                    "preferredquality": "64"}],
                "quiet": True, "no_warnings": True,
            }
            cookies_path = get_youtube_cookies_path()
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            if not os.path.exists(audio_path):
                return None, None
            model = get_whisper_model()
            segments, _ = model.transcribe(audio_path, beam_size=3)
            text = " ".join([s.text.strip() for s in segments])
            return (text, "whisper") if text.strip() else (None, None)
    except Exception:
        return None, None


def get_transcript(video_id):
    """Try all transcript strategies in order."""
    text, method = get_transcript_from_subtitles(video_id)
    if text:
        return text, method
    return get_transcript_from_whisper(video_id)


# ---------------------------------------------------------------------------
# Step 3 — Chunk text
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
    ]


# ---------------------------------------------------------------------------
# Step 4 — Generate embeddings (FREE — runs locally via onnxruntime)
# ---------------------------------------------------------------------------

def get_embedding(text):
    """
    Embed text using ChromaDB's built-in DefaultEmbeddingFunction.
    Uses all-MiniLM-L6-v2 via onnxruntime — no API key, no extra installs,
    and compatible with Python 3.14+.
    """
    fn = get_embed_fn()
    return fn([text])[0]


# ---------------------------------------------------------------------------
# Step 5 — Index an entire channel
# ---------------------------------------------------------------------------

def index_channel(channel_url, max_videos=100, progress_callback=None):
    """Fetch → transcribe → embed → store every video in the channel."""
    collection = get_chroma_collection()

    if progress_callback:
        progress_callback("Fetching video list from channel...")

    videos = get_channel_video_ids(channel_url, max_videos=max_videos)

    if progress_callback:
        progress_callback(f"Found {len(videos)} videos. Starting transcription...")

    indexed_count = 0
    method_counts = {}

    for idx, (video_id, title) in enumerate(videos):
        transcript, method = get_transcript(video_id)

        if not transcript:
            if progress_callback:
                progress_callback(
                    f"[{idx+1}/{len(videos)}] Skipped (no subtitles): {title}"
                )
            continue

        method_counts[method] = method_counts.get(method, 0) + 1
        chunks = chunk_text(transcript)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{video_id}_chunk_{chunk_idx}"
            if collection.get(ids=[doc_id])["ids"]:
                continue
            embedding = get_embedding(chunk)
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
        summary = ", ".join(f"{c} via {m}" for m, c in method_counts.items())
        progress_callback(
            f"\nDone! Indexed {indexed_count}/{len(videos)} videos ({summary})."
        )

    return indexed_count


# ---------------------------------------------------------------------------
# Step 6 — Query the indexed channel
# ---------------------------------------------------------------------------

def query_channel(question, n_results=5):
    """Similarity search + free Groq LLM answer generation."""
    collection = get_chroma_collection()

    if collection.count() == 0:
        return {
            "answer": "No videos indexed yet. Please index a channel first.",
            "sources": [],
        }

    # Embed the question locally (free)
    q_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = [
        f'[Video: "{m["title"]}"]\n{d}'
        for d, m in zip(documents, metadatas)
    ]
    context = "\n\n---\n\n".join(context_parts)

    seen_urls = set()
    sources = []
    for meta in metadatas:
        if meta["url"] not in seen_urls:
            seen_urls.add(meta["url"])
            sources.append({"title": meta["title"], "url": meta["url"]})

    prompt = f"""You are a research assistant that answers questions using YouTube
transcript data. Follow these rules strictly:

1. ONLY use information from the transcript context below — do not add outside knowledge.
2. Use the creator's own words and phrasing where possible. Quote short phrases
   directly (in quotation marks) when they capture the idea well.
3. For EVERY claim or point, cite the specific video title it came from in
   parentheses, e.g. (from "Video Title").
4. If the creator explains the topic across multiple videos, synthesize the
   information and note which video each part comes from.
5. If the context does not contain enough information to answer, say so clearly.
6. Keep the answer well-structured: use short paragraphs, not bullet points.

Transcript context:
{context}

Question: {question}

Answer:"""

    # Answer using Groq's free LLM (llama-3.3-70b — fast and free)
    groq_client = get_groq_client()
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
