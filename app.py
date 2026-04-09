"""
app.py — Streamlit UI for YouTube Channel RAG

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from rag_engine import index_channel, query_channel

# Load .env so OPENAI_API_KEY is available
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="YouTube Channel RAG", page_icon="🎬")
st.title("🎬 YouTube Channel RAG")
st.caption("Turn any YouTube channel into a searchable knowledge base.")

# ---------------------------------------------------------------------------
# Keep a running chat history in session state
# ---------------------------------------------------------------------------
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # list of {"question": ..., "answer": ..., "sources": ...}

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ---------------------------------------------------------------------------
# Section 1 — Index a channel
# ---------------------------------------------------------------------------
st.header("1. Index a Channel")

channel_url = st.text_input("Channel URL", value="", placeholder="Paste a YouTube channel URL here")
max_videos = st.slider("Max videos to index", min_value=5, max_value=500, value=50, step=5)

if st.button("Index Channel"):
    if not channel_url:
        st.warning("Please enter a channel URL first.")
    else:
        status = st.empty()
        progress_log = []

        def on_progress(message):
            progress_log.append(message)
            status.text("\n".join(progress_log[-10:]))

        with st.spinner("Indexing channel — this may take a few minutes..."):
            try:
                count = index_channel(
                    channel_url,
                    max_videos=max_videos,
                    progress_callback=on_progress,
                )
                st.session_state.indexed = True
                st.success(f"Done! Indexed {count} videos.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Ask questions (supports multiple queries)
# ---------------------------------------------------------------------------
st.header("2. Ask Questions")

question = st.text_input("Your question", value="", placeholder="Type any question about the channel's content")

if st.button("Ask"):
    if not question:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching across videos..."):
            try:
                result = query_channel(question)

                # Save to history so all Q&A pairs stay visible
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# Display all past Q&A pairs (newest first)
# ---------------------------------------------------------------------------
if st.session_state.qa_history:
    st.divider()
    st.subheader("Conversation")

    for i, entry in enumerate(reversed(st.session_state.qa_history)):
        with st.container():
            st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {entry['question']}")
            st.write(entry["answer"])
            if entry["sources"]:
                st.markdown("**Sources:**")
                for src in entry["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']})")
            st.divider()

    # Let users clear history if they want a fresh start
    if st.button("Clear History"):
        st.session_state.qa_history = []
        st.rerun()
