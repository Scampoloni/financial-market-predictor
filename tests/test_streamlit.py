"""Smoke tests for Streamlit app entrypoint."""

from pathlib import Path


def test_app_entrypoint_exists() -> None:
	"""Main Streamlit entrypoint file should exist."""
	assert Path("app.py").exists()


def test_app_contains_streamlit_bootstrap() -> None:
	"""Entrypoint should configure Streamlit and declare page config."""
	content = Path("app.py").read_text(encoding="utf-8")
	assert "import streamlit as st" in content
	assert "st.set_page_config(" in content
