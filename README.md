# YouTube Channel RAG

Turn any YouTube channel into a searchable knowledge base. Ask questions across hundreds of videos without watching a single one.

## Project Structure

```
youtube-channel-rag/
├── app.py              # Streamlit UI
├── rag_engine.py       # Backend: fetch, embed, store, query
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key
└── README.md           # You are here
```

## How It Works

1. **Fetch** — `yt-dlp` extracts all video IDs from a YouTube channel
2. **Transcribe** — `youtube-transcript-api` downloads auto-generated captions
3. **Chunk** — Transcripts are split into 500-word overlapping chunks
4. **Embed** — Each chunk is embedded with OpenAI `text-embedding-3-small`
5. **Store** — Embeddings + text are stored in ChromaDB (persistent on disk)
6. **Query** — Your question is embedded, ChromaDB finds the top-5 similar chunks, and GPT-4o-mini generates a grounded answer citing sources

## Run Locally

### 1. Clone and install

```bash
cd youtube-channel-rag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your real key
```

Or export directly:

```bash
export OPENAI_API_KEY=sk-your-key-here
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
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy**

> **Note:** ChromaDB uses local disk storage. On Streamlit Cloud the index resets on each reboot. For production, swap to a hosted vector DB (Pinecone, Weaviate, etc.).

## Example Queries

- "What does this creator say about attention mechanisms?"
- "Summarize all videos about LangChain"
- "Find every mention of RAG"

## What to Build Next

- Add video thumbnails and timestamps to citations
- Filter by date range
- Support playlist URLs
- Compare views across multiple channels
