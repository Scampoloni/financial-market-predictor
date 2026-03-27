"""
rag_chatbot.py — Retrieval-Augmented Generation (RAG) pipeline for financial news Q&A.

Architecture:
  1. Index:  All collected news headlines are embedded using sentence-transformers
             (all-MiniLM-L6-v2, lightweight, runs on CPU in ~1s per query).
  2. Retrieve: For a user query, cosine similarity retrieves the top-k most
               relevant headlines from the index.
  3. Generate: A structured answer is generated using the retrieved context.
               If a Gemini or OpenAI API key is available, it uses the LLM.
               Otherwise, a deterministic template-based fallback is used —
               ensuring the RAG pipeline always works without an API key.

The index is built once from the saved news parquet files and cached in
data/processed/rag_index.pkl. Subsequent loads are instant.

Usage:
    from src.nlp.rag_chatbot import FinancialRAG
    rag = FinancialRAG()
    result = rag.query("What did analysts say about Apple earnings?", ticker="AAPL")
    print(result["answer"])
    print(result["sources"])   # list of retrieved headlines with metadata
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_NEWS_DIR     = _ROOT / "data" / "raw" / "news"
_INDEX_CACHE  = _ROOT / "data" / "processed" / "rag_index.pkl"

# ── Embedding model ───────────────────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 22M params, CPU-friendly, 384-dim


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector a and matrix b (N, D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (b / b_norms) @ a_norm


def _load_embed_model():
    """Lazy-load the sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_EMBED_MODEL_NAME)
        logger.info("SentenceTransformer '%s' loaded.", _EMBED_MODEL_NAME)
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers\n"
            "Falling back to keyword search."
        )
        return None


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(
    news_dir: Path = _NEWS_DIR,
    output_path: Path = _INDEX_CACHE,
    force: bool = False,
) -> dict:
    """Build a vector index of all news headlines.

    Loads all per-ticker parquet files, embeds headlines via sentence-transformers,
    and saves the index to disk.

    Args:
        news_dir: Directory containing {ticker}.parquet news files.
        output_path: Path to save the pickled index.
        force: If True, rebuild even if cache exists.

    Returns:
        Index dict with keys: 'headlines', 'embeddings', 'metadata'.
    """
    if not force and output_path.exists():
        logger.info("Loading RAG index from cache: %s", output_path)
        with open(output_path, "rb") as f:
            return pickle.load(f)

    logger.info("Building RAG index from news files in %s ...", news_dir)
    all_frames = []

    if news_dir.exists():
        for parquet_file in sorted(news_dir.glob("*.parquet")):
            ticker = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                if df.empty or "title" not in df.columns:
                    continue
                df["ticker"] = ticker
                all_frames.append(df[["title", "published", "source", "ticker"]])
            except Exception as exc:
                logger.warning("Could not load %s: %s", parquet_file, exc)

    if not all_frames:
        logger.warning("No news files found at %s — RAG index will be empty.", news_dir)
        return {"headlines": [], "embeddings": np.zeros((0, 384)), "metadata": []}

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.dropna(subset=["title"])
    combined = combined.drop_duplicates(subset=["title"])
    logger.info("Total headlines for indexing: %d", len(combined))

    # Embed with sentence-transformers
    embed_model = _load_embed_model()
    if embed_model is not None:
        embeddings = embed_model.encode(
            combined["title"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    else:
        # TF-IDF fallback: simple bag-of-words vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=512, stop_words="english")
        embeddings = vec.fit_transform(combined["title"].tolist()).toarray().astype(np.float32)

    headlines = combined["title"].tolist()
    metadata  = combined.to_dict("records")

    index = {"headlines": headlines, "embeddings": embeddings, "metadata": metadata}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
    logger.info("RAG index saved to %s (%d headlines, embed_dim=%d)",
                output_path, len(headlines), embeddings.shape[1])
    return index


# ── Retriever ─────────────────────────────────────────────────────────────────

class FinancialRAG:
    """Retrieval-Augmented Generation over collected financial news headlines.

    Supports optional ticker filtering, LLM answer generation (Gemini / OpenAI),
    and a deterministic template-based fallback when no API key is configured.

    Args:
        index_path: Path to cached index pickle.
        news_dir: Path to raw news parquet files.
        top_k: Number of headlines to retrieve per query.
    """

    def __init__(
        self,
        index_path: Path = _INDEX_CACHE,
        news_dir: Path = _NEWS_DIR,
        top_k: int = 5,
    ) -> None:
        self._index_path  = index_path
        self._news_dir    = news_dir
        self._top_k       = top_k
        self._index: dict | None = None
        self._embed_model = None

    def _ensure_index(self) -> None:
        if self._index is None:
            self._index = build_index(self._news_dir, self._index_path)
            self._embed_model = _load_embed_model()

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve top-k most relevant headlines for a query.

        Args:
            query: Natural language question.
            ticker: If provided, restrict results to this ticker only.
            top_k: Override default top_k.

        Returns:
            List of result dicts, each with keys:
            'headline', 'ticker', 'published', 'source', 'score'.
        """
        self._ensure_index()
        k = top_k or self._top_k

        embeddings = self._index["embeddings"]
        metadata   = self._index["metadata"]

        if len(metadata) == 0:
            return []

        # Ticker filter: mask out non-matching rows
        if ticker:
            ticker_upper = ticker.upper()
            mask = np.array([m.get("ticker", "").upper() == ticker_upper for m in metadata])
            if mask.sum() == 0:
                logger.info("No headlines found for ticker %s — using all.", ticker)
                mask = np.ones(len(metadata), dtype=bool)
            filtered_embeds = embeddings[mask]
            filtered_meta   = [m for m, keep in zip(metadata, mask) if keep]
        else:
            filtered_embeds = embeddings
            filtered_meta   = metadata

        # Encode query
        if self._embed_model is not None:
            query_vec = self._embed_model.encode([query], convert_to_numpy=True)[0]
        else:
            # TF-IDF keyword fallback: use term overlap
            query_lower = query.lower()
            query_vec = np.zeros(filtered_embeds.shape[1])
            for i, word in enumerate(query_lower.split()):
                query_vec[i % len(query_vec)] += 1.0

        scores = _cosine_sim(query_vec, filtered_embeds)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            m = filtered_meta[idx]
            results.append({
                "headline": m.get("title", ""),
                "ticker":   m.get("ticker", ""),
                "published": str(m.get("published", "")),
                "source":   m.get("source", ""),
                "score":    float(scores[idx]),
            })
        return results

    def query(
        self,
        question: str,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Answer a financial question using retrieved news context.

        Args:
            question: User's natural language question.
            ticker: Optional ticker symbol to focus on.
            top_k: Number of headlines to retrieve.

        Returns:
            Dict with 'answer' (str) and 'sources' (list[dict]).
        """
        sources = self.retrieve(question, ticker=ticker, top_k=top_k)

        # Try LLM answer generation (Gemini preferred, OpenAI fallback)
        answer = self._generate_llm_answer(question, sources, ticker)
        if answer is None:
            answer = self._template_answer(question, sources, ticker)

        return {"answer": answer, "sources": sources}

    def _generate_llm_answer(
        self,
        question: str,
        sources: list[dict],
        ticker: Optional[str],
    ) -> str | None:
        """Try to generate an answer using an available LLM API.

        Returns None if no API key is configured.
        """
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        # Build context from retrieved headlines
        context = "\n".join(
            f"[{s['ticker']} | {s['published'][:10]}] {s['headline']}"
            for s in sources
        )
        ticker_clause = f" for {ticker}" if ticker else ""
        system_prompt = (
            "You are a financial news analyst assistant. "
            "Answer the user's question using ONLY the provided news headlines as context. "
            "Be concise (2-4 sentences). If the context doesn't answer the question, say so. "
            "Never make up news or facts not in the context."
        )
        user_prompt = (
            f"Question{ticker_clause}: {question}\n\n"
            f"Context headlines:\n{context}\n\n"
            "Answer based only on the above headlines:"
        )

        # Try Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
                return response.text.strip()
            except Exception as exc:
                logger.warning("Gemini API error: %s", exc)

        # Try OpenAI fallback
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=300,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI API error: %s", exc)

        return None  # No LLM available

    def _template_answer(
        self,
        question: str,
        sources: list[dict],
        ticker: Optional[str],
    ) -> str:
        """Generate a deterministic template-based answer without an LLM.

        Always works — no API key required.
        """
        if not sources:
            scope = f" for {ticker}" if ticker else ""
            return (
                f"No relevant news headlines found{scope} matching your question. "
                "The news index may be empty or the query may be too specific."
            )

        top_headlines = [s["headline"] for s in sources[:3]]
        scope = f" about {ticker}" if ticker else ""
        date_range = sources[0]["published"][:10] if sources else "recently"

        answer = (
            f"Based on {len(sources)} retrieved headlines{scope} "
            f"(most recent around {date_range}), here are the most relevant findings:\n\n"
        )
        for i, hl in enumerate(top_headlines, 1):
            answer += f"{i}. {hl}\n"

        answer += (
            "\n💡 *Tip: Provide a Gemini or OpenAI API key in your .env to get "
            "natural language summaries instead of raw headlines.*"
        )
        return answer

    def rebuild_index(self) -> None:
        """Force-rebuild the vector index from scratch."""
        self._index = build_index(self._news_dir, self._index_path, force=True)
