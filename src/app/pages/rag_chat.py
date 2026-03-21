"""rag_chat.py — News chatbot interface (RAG Q&A page) — placeholder."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("News Q&A Chatbot (RAG)")
    st.info(
        "The RAG chatbot is implemented in a separate feature branch (`feature/rag-chatbot`) "
        "and will be merged in a future release. "
        "It will allow natural-language questions over the collected news headlines "
        "using a retrieval-augmented generation pipeline."
    )
    st.markdown("""
**Planned functionality:**
- Vector store of all collected news headlines (FAISS + sentence-transformers)
- Retrieval of top-k relevant articles for a user query
- Answer generation via a language model (Claude API)
- Ticker-filtered Q&A: "What did analysts say about NVDA earnings last week?"
    """)
