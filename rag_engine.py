"""
rag_engine.py — Core backend for YouTube Channel RAG

Transcript strategy (all FREE, 403-resistant):
  1. yt-dlp subtitle download — downloads .vtt files directly, handles
     YouTube bot-detection far better than youtube-transcript-api.
     Tries English first, then any available language.
  2. Optional cookies — if YOUTUBE_COOKIES is set in Streamlit secrets
     (or the env), it's passed to yt-dlp for maximum reliability on
     cloud IPs where YouTube is most aggressive about blocking.
"""

import os
import re
import tempfile
from yt_dlp import YoutubeDL
from openai import OpenAI
import chromadb

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

def get_openai_client():
    """Return OpenAI client — checks Streamlit secrets then env variable."""
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "For local: add to .env. "
            "For Streamlit Cloud: add in App Settings → Secrets."
        )
    return OpenAI(api_key=api_key)


def get_chroma_collection(collection_name="youtube_channel"):
    """Return (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name=collection_name)


def get_youtube_cookies_path():
    """
    If YOUTUBE_COOKIES is set (Netscape/cookies.txt format), write it to a
    temp file and return the path so yt-dlp can use it.

    How to get your cookies:
      1. Install the browser extension "Get cookies.txt LOCALLY"
      2. Open youtube.com while logged in
      3. Click the extension → Export → copy the full text
      4. Paste it into Streamlit Cloud Secrets as:
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

    # Write to a temp file — yt-dlp reads from disk
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
    """
    Extract video IDs from a YouTube channel URL using yt-dlp.
    Normalises any channel URL format to the /videos tab.
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

    cookies_path = get_youtube_cookies_path()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

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
# Step 2 — Fetch subtitles via yt-dlp (replaces youtube-transcript-api)
# ---------------------------------------------------------------------------

def parse_vtt(vtt_text):
    """
    Convert a WebVTT subtitle string to plain text.
    Strips timing lines, HTML tags, and deduplicates repeated lines
    (auto-generated captions often repeat the same line in consecutive cues).
    """
    text_lines = []
    for line in vtt_text.splitlines():
        line = line.strip()
        # Skip header, cue timings, blank lines, and cue identifiers
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("NOTE") or line.startswith("STYLE"):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):          # bare cue numbers
            continue
        # Strip inline tags: <00:00:00.000>, <c>, </c>, <b>, etc.
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if line:
            text_lines.append(line)

    # Remove consecutive duplicates (very common in auto-captions)
    deduped = []
    prev = None
    for line in text_lines:
        if line != prev:
            deduped.append(line)
            prev = line

    return " ".join(deduped)


def get_transcript_from_subtitles(video_id):
    """
    Download subtitles for a video using yt-dlp and return the text.

    yt-dlp handles YouTube's bot-detection far better than direct HTTP
    requests, so this works reliably on Streamlit Cloud where
    youtube-transcript-api gets 403 errors.

    Tries in order:
      1. Manual English subtitles
      2. Auto-generated English subtitles
      3. Any other language (auto or manual)
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,       # manual captions
                "writeautomaticsub": True,    # auto-generated captions
                "subtitlesformat": "vtt",
                # Try English first, then a broad set of common languages
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

            # Collect all downloaded .vtt files
            vtt_files = [
                f for f in os.listdir(tmp_dir) if f.endswith(".vtt")
            ]
            if not vtt_files:
                return None, None

            # Prefer English subtitle files
            en_files = [f for f in vtt_files if ".en" in f.lower()]
            chosen = en_files[0] if en_files else vtt_files[0]
            lang_label = "en" if en_files else chosen.split(".")[-2]

            with open(os.path.join(tmp_dir, chosen), "r", encoding="utf-8") as f:
                vtt_text = f.read()

            text = parse_vtt(vtt_text)
            return (text, f"subtitles ({lang_label})") if text.strip() else (None, None)

    except Exception:
        return None, None


def get_transcript_from_whisper(video_id):
    """
    Last-resort fallback: download audio and transcribe with local Whisper.
    Only runs if faster-whisper is installed (won't run on Streamlit Cloud).
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
            cookies_path = get_youtube_cookies_path()
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path

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
    Get a transcript for a video using all available strategies:
      1. yt-dlp subtitle download (manual + auto-generated, any language)
      2. Local Whisper transcription (only if faster-whisper is installed)

    Returns: (transcript_text, method_label) or (None, None)
    """
    text, method = get_transcript_from_subtitles(video_id)
    if text:
        return text, method

    text, method = get_transcript_from_whisper(video_id)
    if text:
        return text, method

    return None, None


# ---------------------------------------------------------------------------
# Step 3 — Chunk text into overlapping pieces
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into word-level chunks with overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Generate embeddings
# ---------------------------------------------------------------------------

def get_embedding(text, client=None):
    """Embed text with OpenAI text-embedding-3-small."""
    if client is None:
        client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step 5 — Index an entire channel
# ---------------------------------------------------------------------------

def index_channel(channel_url, max_videos=100, progress_callback=None):
    """Fetch → transcribe → embed → store every video in the channel."""
    openai_client = get_openai_client()
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
                    f"[{idx+1}/{len(videos)}] Skipped (no subtitles available): {title}"
                )
            continue

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
        summary = ", ".join(f"{c} via {m}" for m, c in method_counts.items())
        progress_callback(
            f"\nDone! Indexed {indexed_count}/{len(videos)} videos ({summary})."
        )

    return indexed_count


# ---------------------------------------------------------------------------
# Step 6 — Query the indexed channel
# ---------------------------------------------------------------------------

def query_channel(question, n_results=5):
    """Similarity search + GPT answer generation."""
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
