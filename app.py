"""Voice-enabled RAG demo over MS MARCO-XI (Hindi).

Run:  streamlit run app.py        (needs data/index/ built first)

Input paths: record via browser mic -> Sarvam STT; upload an audio file ->
Sarvam STT; or type a question (no STT). Answers come from the grounded
extractive generator (citations always shown); the optional fluent-LLM rung
turns on with VRAG_LLM=1. Guardrail refusals are shown as such, not as
answers.
"""

import json
import os
from pathlib import Path

import streamlit as st

from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Voice RAG · MS MARCO-XI", layout="wide")

from voice_rag.config import get_settings
from voice_rag.harness import Status, pipeline_from_artifacts

INDEX_DIR = Path(os.getenv("VRAG_INDEX", "data/index"))


@st.cache_resource(show_spinner="Loading index + embedding model (first run only)…")
def _pipeline():
    return pipeline_from_artifacts(INDEX_DIR)


STATUS_COPY = {
    Status.ANSWERED: ("Answer", "ok"),
    Status.BLOCKED: ("Blocked", "err"),
    Status.OFF_TOPIC: ("Outside knowledge base", "warn"),
    Status.LOW_CONFIDENCE: ("Low confidence", "warn"),
    Status.ERROR: ("Error", "err"),
    Status.NO_ANSWER: ("No answer", "warn"),
}


def _render_result(res):
    title, tone = STATUS_COPY.get(res.status, ("Result", "warn"))
    color = {"ok": "green", "warn": "orange", "err": "red"}[tone]
    st.markdown(f"### <span style='color:{color}'>{title}</span>",
                unsafe_allow_html=True)
    if res.status is Status.ANSWERED:
        st.markdown(f"**{res.answer}**")
        if res.generator:
            st.caption(f"generator: `{res.generator}` · confidence "
                       f"`{res.confidence}` · intent `{res.intent}`")
        if res.citations:
            with st.expander(f"Cited passages ({len(res.citations)})"):
                for i, c in enumerate(res.citations, 1):
                    st.markdown(
                        f"{i}. **{c.passage_id}** (sim={c.vector_sim}, "
                        f"bm25={c.bm25_score})  \n{c.text}")
    elif res.blocked_reason:
        st.warning(res.blocked_reason)
    if res.transcript:
        st.caption(f"transcribed: {res.transcript}")
    if res.stage_ms:
        parts = " · ".join(f"{k}={v}ms" for k, v in res.stage_ms.items())
        st.caption(f"latency: {parts}  (total {res.total_ms}ms)")


def main():
    s = get_settings()
    st.title("Voice RAG — MS MARCO-XI (हिन्दी)")
    st.caption(
        "Speak a question (Sarvam STT) or type it. Retrieval is hybrid "
        "semantic + BM25 over a chunked passage index; answers are grounded in "
        "the retrieved context and refused when they aren't.")

    with st.sidebar:
        st.header("Input")
        mode = st.radio("How do you want to ask?",
                        ["Record audio", "Upload audio", "Type a question"])
        query_text = None
        audio = None
        ctype = "audio/wav"
        if mode == "Record audio":
            audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="Stop",
                                 format="wav", key="mic")
        elif mode == "Upload audio":
            up = st.file_uploader("Audio (wav/mp3/ogg/…)", type=None)
            if up:
                audio = up.getvalue()
                ctype = up.type or "audio/wav"
        else:
            query_text = st.text_input("Question (Hindi or English)")
        run = st.button("Answer", type="primary", use_container_width=True)
        st.divider()
        llm_on = os.getenv("VRAG_LLM", "0") == "1"
        stt_chain = s.stt_fallbacks or []
        chain_label = " → ".join([s.stt_provider] + stt_chain)
        st.caption(f"STT chain: `{chain_label}` "
                   f"({'(Sarvam key set)' if s.sarvam_api_key else '(no key — vosk fallback active)'}) "
                   f"· fluent-LLM rung: {'on' if llm_on else 'off'}")

    if not INDEX_DIR.exists():
        st.error("Index not built. Run `python scripts/build_index.py` first.")
        st.stop()
    pipe = _pipeline()

    if run:
        try:
            if query_text:
                res = pipe.run_from_text(query_text)
            elif audio:
                res = pipe.run_from_audio(audio, ctype)
            else:
                st.info("Record/upload audio or type a question first.")
                st.stop()
        except Exception as e:  # noqa: BLE001 — never let the demo crash
            st.error(f"Pipeline error: {type(e).__name__}: {e}")
            st.stop()
        st.divider()
        _render_result(res)

    st.divider()
    st.caption("Guardrails: input/safety screen · off-topic gate · "
               "retrieval-confidence gate · hallucination check for LLM "
               "answers. Latency analytics: `scripts/benchmark.py`.")


if __name__ == "__main__":
    main()