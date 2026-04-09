# YouTube Channel RAG

Turn any YouTube channel into a searchable knowledge base. Ask questions across hundreds of videos without watching a single one.

## Project Structure

```
youtube-channel-rag/
├── app.py              # Streamlit UI
├── rag_engine.py       # Backend: fetch, embed, store, query
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (ffmpeg) for Streamlit Cloud
├── .env.example        # Template for your API key
└── README.md           # You are here
```

## How It Works

1. **Fetch** — `yt-dlp` extracts all video IDs from a YouTube channel
2. **Transcribe** — `yt-dlp` downloads subtitle files (.vtt) for each video (manual or auto-generated, any language)
3. **Chunk** — Transcripts are split into 500-word overlapping chunks
4. **Embed** — Each chunk is embedded locally using ChromaDB's built-in `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via onnxruntime)
5. **Store** — Embeddings + text are stored in ChromaDB (persistent on disk)
6. **Query** — Your question is embedded, ChromaDB finds the top-5 similar chunks, and Groq's Llama 3.3 70B generates a grounded answer citing sources

## Tech Stack (100% Free)

| Component | Tool | Cost |
|---|---|---|
| Video listing | yt-dlp | Free |
| Transcripts | yt-dlp subtitle download | Free |
| Embeddings | ChromaDB DefaultEmbeddingFunction (onnxruntime) | Free (local) |
| Vector DB | ChromaDB | Free (local) |
| LLM | Groq API — Llama 3.3 70B | Free tier (14,400 req/day) |
| UI | Streamlit | Free |

## Run Locally

### 1. Clone and install

```bash
cd youtube-channel-rag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com)

```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=gsk_your_key_here
```

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as the main file
4. In **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**

> **Note:** ChromaDB uses local disk storage. On Streamlit Cloud the index resets on each reboot. For production, swap to a hosted vector DB (Pinecone, Weaviate, etc.).

## Example Queries

- "What does this creator say about attention mechanisms?"
- "Summarize all videos about LangChain"
- "Find every mention of RAG"

---

## Problems I Faced & How I Solved Them

Building this project for Streamlit Cloud deployment surfaced several real-world issues. Here's every problem I hit and exactly how I fixed it:

### 1. OPENAI_API_KEY not found on Streamlit Cloud

**Problem:** The app used `os.environ.get("OPENAI_API_KEY")` which works locally with `.env` files, but Streamlit Cloud doesn't load `.env` — it uses its own Secrets system (`st.secrets`).

**Fix:** Updated `get_openai_client()` to check `st.secrets` first, then fall back to `os.environ`. This way the same code works both locally and on Streamlit Cloud without changes.

### 2. yt-dlp returned channel tabs instead of actual videos

**Problem:** Using `extract_flat: True` with a channel URL like `youtube.com/@Creator` made yt-dlp return 3 entries — "Videos", "Live", "Shorts" — which are the channel's tab names, not actual videos. So every channel showed "Found 3 videos" and all 3 were skipped.

**Fix:** Two changes — (a) auto-append `/videos` to any channel URL so yt-dlp targets the right tab, and (b) switch from `extract_flat: True` to `extract_flat: "in_playlist"` which goes one level deeper into the playlist. Also added a filter to only accept 11-character video IDs, skipping any sub-playlist entries.

### 3. youtube-transcript-api only tried English captions

**Problem:** The original `YouTubeTranscriptApi.get_transcript(video_id)` only looks for English captions. Many YouTube channels (especially Indian creators) have auto-generated captions in Hindi or other languages but not English. Result: "Skipped (no transcript)" for every single video, "Indexed 0 videos."

**Fix:** Rewrote the transcript function to use `list_transcripts()` and try four strategies in order — English manual captions, any manual caption translated to English, English auto-generated captions, and any auto-generated caption translated to English. This caught 95%+ of videos that were previously skipped.

### 4. HTTP 403 Forbidden errors on Streamlit Cloud

**Problem:** Even after fixing the language issue, `youtube-transcript-api` still failed on Streamlit Cloud with `HTTP Error 403: Forbidden` for every video. YouTube actively blocks raw HTTP requests from cloud server IPs (AWS, GCP, etc.) that Streamlit Cloud runs on.

**Fix:** Replaced `youtube-transcript-api` entirely with `yt-dlp`'s built-in subtitle downloader (`writesubtitles` + `writeautomaticsub`). `yt-dlp` is specifically designed to handle YouTube's bot-detection — it mimics a real browser, rotates user-agents, and handles rate limiting. Also added optional YouTube cookie support as a nuclear option if yt-dlp alone isn't enough.

### 5. OpenAI API quota exceeded (429 error)

**Problem:** OpenAI's free trial credits expired. All embedding and GPT calls returned `429 — insufficient_quota`, requiring a paid plan ($5+ minimum top-up).

**Fix:** Replaced the entire OpenAI dependency with free alternatives — `sentence-transformers` (all-MiniLM-L6-v2) for embeddings running locally, and Groq's free API (Llama 3.3 70B, 14,400 requests/day free) for LLM answers. Total cost went from ~$0.50/channel to $0.

### 6. sentence-transformers doesn't support Python 3.14

**Problem:** Streamlit Cloud upgraded to Python 3.14.3, but `sentence-transformers` hasn't released a compatible build yet. The app crashed on import with `ModuleNotFoundError: No module named 'sentence_transformers'`.

**Fix:** Replaced `sentence-transformers` with ChromaDB's built-in `DefaultEmbeddingFunction`, which uses the exact same all-MiniLM-L6-v2 model but runs it through `onnxruntime` instead. Works on Python 3.14+, no extra installs needed beyond `onnxruntime` in requirements.txt.

### 7. faster-whisper doesn't build on Streamlit Cloud

**Problem:** I added `faster-whisper` as a fallback to transcribe videos that have no subtitles at all (by downloading audio and running Whisper locally). But `faster-whisper` depends on `ctranslate2` which fails to build on Streamlit Cloud's environment.

**Fix:** Made `faster-whisper` an optional dependency — wrapped the import in `try/except` so the app starts normally without it. On Streamlit Cloud, the Whisper fallback is simply skipped. Locally, users can `pip install faster-whisper` to enable it for videos that truly have no subtitles.

---

## What to Build Next

- Add video thumbnails and timestamps to citations
- Filter by date range ("only videos from the last 3 months")
- Support playlist URLs in addition to full channels
- Add a comparison mode: "How does Creator A's view on X differ from Creator B?"
- Swap to a hosted vector DB (Pinecone/Weaviate) for persistent cloud storage
