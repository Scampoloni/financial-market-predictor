"""rag_chat.py — News Q&A chatbot powered by RAG (Retrieval-Augmented Generation)."""

from __future__ import annotations

import streamlit as st
from html import escape as html_escape

_MUTED = "#64748b"
_ACCENT = "#8b5cf6"
_UP_COLOR = "#10b981"


@st.cache_resource(show_spinner="Building news index...")
def _get_rag():
    """Load and cache the FinancialRAG instance (index built once)."""
    from src.nlp.rag_chatbot import FinancialRAG
    rag = FinancialRAG()
    rag._ensure_index()
    return rag


def render() -> None:
    st.markdown("<h2 style='margin-bottom:4px'>News Q&A Chatbot</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.9rem;margin-bottom:1.2rem'>"
        "Ask questions about financial news headlines. Retrieval-Augmented Generation (RAG) "
        "finds the most relevant headlines and generates a grounded answer.</p>",
        unsafe_allow_html=True,
    )

    # ── RAG method explainer ──────────────────────────────────────────────────
    with st.expander("ℹ️ How this works", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="glass-card" style="text-align:center">'
                '<div style="font-size:1.4rem;margin-bottom:6px">🔍</div>'
                '<div style="color:#8b5cf6;font-weight:700;margin-bottom:6px">1. Retrieve</div>'
                '<div style="color:#94a3b8;font-size:0.83rem">'
                'Your query is embedded using <b style="color:#e2e8f0">all-MiniLM-L6-v2</b> '
                '(sentence-transformers). Cosine similarity finds the top-k most '
                'relevant headlines from ~6,000 indexed news articles.</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="glass-card" style="text-align:center">'
                '<div style="font-size:1.4rem;margin-bottom:6px">📄</div>'
                '<div style="color:#f59e0b;font-weight:700;margin-bottom:6px">2. Augment</div>'
                '<div style="color:#94a3b8;font-size:0.83rem">'
                'Retrieved headlines are bundled as context. '
                'The prompt explicitly instructs the model to answer '
                '<b style="color:#e2e8f0">only from the provided sources</b> — '
                'no hallucinated facts.</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="glass-card" style="text-align:center">'
                '<div style="font-size:1.4rem;margin-bottom:6px">💬</div>'
                '<div style="color:#10b981;font-weight:700;margin-bottom:6px">3. Generate</div>'
                '<div style="color:#94a3b8;font-size:0.83rem">'
                'Gemini 1.5 Flash (or OpenAI GPT-4o-mini if configured) generates '
                'the answer. Without an API key, the retrieved headlines are '
                '<b style="color:#e2e8f0">shown directly</b>.</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

    # ── Query controls ────────────────────────────────────────────────────────
    col_q, col_t, col_k = st.columns([4, 1.5, 1])

    with col_q:
        question = st.text_input(
            "Your question",
            placeholder='e.g. "What did analysts say about NVIDIA\'s AI growth?"',
            label_visibility="collapsed",
            key="rag_question",
        )

    with col_t:
        ticker_filter = st.text_input(
            "Ticker filter (optional)",
            placeholder="e.g. NVDA",
            label_visibility="collapsed",
            key="rag_ticker",
        ).strip().upper() or None

    with col_k:
        top_k = st.selectbox(
            "# Sources", [3, 5, 8, 10],
            index=1, label_visibility="collapsed", key="rag_topk",
        )

    # Example questions
    st.markdown(
        f"<div style='margin:-0.3rem 0 0.6rem;color:{_MUTED};font-size:0.82rem'>"
        "<b>Examples:</b> "
        '"What happened with Fed interest rates?" · '
        '"Show me negative news about Tesla" · '
        '"What are analyst upgrades for tech stocks?"'
        "</div>",
        unsafe_allow_html=True,
    )

    col_ask, col_live = st.columns([1, 4])
    with col_ask:
        ask_btn = st.button("Ask", type="primary", use_container_width=False)
    with col_live:
        fetch_live_btn = False
        if ticker_filter:
            fetch_live_btn = st.button("🔄 Fetch Live News for Ticker", help=f"Downloads the latest news for {ticker_filter} into the index.")
            
    if fetch_live_btn and ticker_filter:
        with st.spinner(f"Fetching latest news for {ticker_filter}..."):
            try:
                from src.data_collection.news_scraper import collect_all
                from src.features.nlp_features import update_single_ticker_nlp
                collect_all([ticker_filter])
                update_single_ticker_nlp(ticker_filter)
                rag = _get_rag()
                rag.rebuild_index()
                st.cache_resource.clear()
                st.success(f"Successfully fetched new headlines for {ticker_filter} and rebuilt index!")
            except Exception as e:
                st.error(f"Failed to fetch live news: {e}")

    # ── Chat history (session state) ──────────────────────────────────────────
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []  # list of {question, answer, sources}

    if ask_btn and question.strip():
        with st.spinner("Searching news index and generating answer…"):
            try:
                rag = _get_rag()
                result = rag.query(question.strip(), ticker=ticker_filter, top_k=top_k)
                st.session_state.rag_history.insert(0, {
                    "question": question.strip(),
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "ticker": ticker_filter,
                })
            except Exception as exc:
                st.error(f"RAG error: {exc}")

    if st.session_state.rag_history:
        if st.button("Clear history", type="secondary", key="rag_clear"):
            st.session_state.rag_history = []
            st.rerun()

    # ── Render history ────────────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.rag_history):
        _render_qa_entry(entry, index=i)

    if not st.session_state.rag_history and not (ask_btn and question.strip()):
        st.markdown(
            f"<p style='color:{_MUTED};text-align:center;margin-top:2rem;font-size:0.95rem'>"
            "Type a question above to search the news database.</p>",
            unsafe_allow_html=True,
        )

    # ── Rebuild index button ──────────────────────────────────────────────────
    with st.expander("🔧 Index management", expanded=False):
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.85rem'>"
            "The RAG index is built automatically from all collected news headlines. "
            "Rebuild if you've added new news data.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Rebuild news index", key="rag_rebuild"):
            with st.spinner("Rebuilding index…"):
                try:
                    rag = _get_rag()
                    rag.rebuild_index()
                    st.cache_resource.clear()
                    st.success("Index rebuilt successfully!")
                except Exception as exc:
                    st.error(f"Error rebuilding index: {exc}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='disclaimer'>RAG answers are based solely on collected news headlines. "
        "Not financial advice. Verify all information independently.</p>",
        unsafe_allow_html=True,
    )


def _render_qa_entry(entry: dict, index: int) -> None:
    """Render a single Q&A exchange with sources."""
    ticker_badge = ""
    if entry.get("ticker"):
        ticker_badge = (
            f'<span style="background:#172554;color:#60a5fa;padding:2px 10px;'
            f'border-radius:20px;font-size:0.78rem;font-weight:700;'
            f'margin-left:8px">{html_escape(entry["ticker"])}</span>'
        )

    # Question bubble
    st.markdown(
        f'<div style="background:#1e293b;border-radius:12px;padding:12px 16px;'
        f'margin-bottom:8px;border-left:3px solid #8b5cf6">'
        f'<span style="color:#94a3b8;font-size:0.82rem;font-weight:600">Q:</span>'
        f'{ticker_badge}'
        f'<div style="color:#f0f6fc;font-size:0.95rem;margin-top:4px">'
        f'{html_escape(entry["question"])}</div></div>',
        unsafe_allow_html=True,
    )

    # Answer bubble
    import re
    safe_answer = html_escape(entry["answer"]).replace("\n", "<br>")
    # Restore bold markers that were escaped (e.g. *Tip:*)
    safe_answer = re.sub(r"\*(.+?)\*", r"<em>\1</em>", safe_answer)

    st.markdown(
        f'<div class="glass-card" style="border-left:3px solid #10b981;margin-bottom:10px">'
        f'<span style="color:#94a3b8;font-size:0.82rem;font-weight:600">A:</span>'
        f'<div style="color:#e2e8f0;font-size:0.92rem;margin-top:6px;line-height:1.7">'
        f'{safe_answer}</div></div>',
        unsafe_allow_html=True,
    )

    # Retrieved sources
    if entry.get("sources"):
        with st.expander(f"📎 {len(entry['sources'])} retrieved sources", expanded=False):
            for src in entry["sources"]:
                score_pct = int(src["score"] * 100)
                score_color = "#10b981" if score_pct >= 50 else "#f59e0b" if score_pct >= 30 else "#64748b"
                st.markdown(
                    f'<div class="headline-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<div style="flex:1;margin-right:12px">'
                    f'<div style="font-weight:500;font-size:0.88rem;color:#e2e8f0">'
                    f'{html_escape(src["headline"])}</div>'
                    f'<div style="color:#64748b;font-size:0.76rem;margin-top:2px">'
                    f'{html_escape(src["ticker"])} · {html_escape(src["source"])} · {src["published"][:10]}'
                    f'</div></div>'
                    f'<span style="background:{score_color}18;color:{score_color};padding:3px 8px;'
                    f'border-radius:8px;font-size:0.78rem;font-weight:700;white-space:nowrap">'
                    f'{score_pct}% match</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    if index < len(st.session_state.rag_history) - 1:
        st.markdown("<hr>", unsafe_allow_html=True)
