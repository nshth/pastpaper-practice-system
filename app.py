import re
import streamlit as st
from src.llm.intent_classifier import detect_intent
from src.llm.search_questions import search_questions
from src.llm.explain import stream_explanation, stream_general

ASSETS_DIR = "data/extracted/assets"

st.set_page_config(page_title="ICT Past Papers", page_icon="📚", layout="centered")
st.title("ICT Past Paper Assistant")
st.caption("Search questions by unit or year, get explanations, or unlock learning hacks to fix your cooked study habits.")

if "history" not in st.session_state:
    st.session_state.history = []


def resolve_answer(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0]
    return int(raw)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def render_question_card(q: dict, idx: int):
    label = f"Q{q['question_number']} | Unit {q['unit']} – {q.get('unit_name', '?')} | {q['year']} P{q['paper_number']}"
    with st.expander(label):
        text = q["text"]

        # Render images inline, then clean remaining text
        img_pattern = re.compile(r"!\[.*?\]\((images/[^)]+)\)")
        last_end = 0
        parts = []
        for m in img_pattern.finditer(text):
            parts.append(("text", clean_text(text[last_end:m.start()])))
            parts.append(("img", f"{ASSETS_DIR}/{m.group(1)}"))
            last_end = m.end()
        parts.append(("text", clean_text(text[last_end:])))

        for kind, content in parts:
            if kind == "img":
                try:
                    st.image(content)
                except Exception:
                    st.caption(f"[Image not found: {content}]")
            elif content.strip():
                st.write(content)

        correct = resolve_answer(q.get("correct_option"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("idk the answer", key=f"ans_{q['question_id']}_{idx}"):
                st.success(f"Correct answer: Option ({correct})")
        with col2:
            if st.button("Explain like im 5", key=f"exp_{q['question_id']}_{idx}"):
                clean_q = clean_text(q["text"])
                with st.spinner("Explaining..."):
                    full = ""
                    placeholder = st.empty()
                    for token in stream_explanation(clean_q, correct, st.session_state.history):
                        full += token
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                st.session_state.history.append({"role": "assistant", "content": full})
                st.rerun()


def render_history():
    for entry in st.session_state.history:
        role = entry["role"]
        if role == "results":
            with st.chat_message("assistant"):
                st.markdown(entry["summary"])
            for idx, q in enumerate(entry["questions"]):
                render_question_card(q, idx)
        elif role in ("user", "assistant"):
            with st.chat_message(role):
                st.markdown(entry["content"])


render_history()

user_input = st.chat_input("Ask something... e.g. 'show OS questions' or 'explain binary encoding'")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        parsed = detect_intent(user_input)

    intent = parsed.get("intent", "general")

    if intent == "search":
        unit = parsed.get("unit")
        year = parsed.get("year")
        results = search_questions(unit=unit, year=year)
        summary = f"Found **{len(results)}** question(s)"
        if unit:
            summary += f" for Unit {unit}"
        if year:
            summary += f" from {year}"

        st.session_state.history.append({
            "role": "results",
            "summary": summary,
            "questions": results,
        })
        with st.chat_message("assistant"):
            st.markdown(summary)
        for idx, q in enumerate(results):
            render_question_card(q, idx)

    elif intent == "explain":
        question_text = parsed.get("question_text") or user_input
        with st.chat_message("assistant"):
            full = ""
            placeholder = st.empty()
            for token in stream_explanation(question_text, None, st.session_state.history):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.history.append({"role": "assistant", "content": full})

    else:
        with st.chat_message("assistant"):
            full = ""
            placeholder = st.empty()
            for token in stream_general(user_input, st.session_state.history):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.history.append({"role": "assistant", "content": full})