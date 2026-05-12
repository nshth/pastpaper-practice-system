import streamlit as st
from src.llm.intent_classifier import detect_intent
from src.llm.search_questions import search_questions
from src.llm.explain import stream_explanation, stream_general

st.set_page_config(page_title="ICT Past Papers", page_icon="📚", layout="centered")
st.title("ICT Past Paper Assistant")
st.caption("Search questions by unit or year, get explanations, or unlock learning hacks to fix your cooked study habits.")

if "history" not in st.session_state:
    st.session_state.history = []

if "explain_target" not in st.session_state:
    st.session_state.explain_target = None


def render_questions(questions: list[dict]):
    if not questions:
        st.info("No questions found for that filter.")
        return
    for q in questions:
        label = f"Q{q['question_number']} | Unit {q['unit']} – {q.get('unit_name','?')} | {q['year']} P{q['paper_number']}"
        with st.expander(label):
            st.write(q["text"])
            if st.button("idk the answer", key=f"ans_{q['question_id']}"):
                st.success(f"Correct answer: Option ({q['correct_option']})")
            if st.button("Explain like im 5 yr old", key=f"exp_{q['question_id']}"):
                st.session_state.explain_target = q


# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Render search results if present
if "last_results" in st.session_state and st.session_state.last_results:
    render_questions(st.session_state.last_results)

# Handle explain button click from search results
if st.session_state.explain_target:
    q = st.session_state.explain_target
    st.session_state.explain_target = None
    with st.chat_message("assistant"):
        with st.spinner("Explaining..."):
            full = ""
            placeholder = st.empty()
            for token in stream_explanation(q["text"], q.get("correct_option")):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
    st.session_state.history.append({"role": "assistant", "content": full})

# Chat input
user_input = st.chat_input("Ask something... e.g. 'show unit 6 questions' or 'explain this question'")

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
        st.session_state.last_results = results
        summary = f"Found {len(results)} question(s)"
        if unit:
            summary += f" for Unit {unit}"
        if year:
            summary += f" from {year}"
        with st.chat_message("assistant"):
            st.markdown(summary)
        st.session_state.history.append({"role": "assistant", "content": summary})
        render_questions(results)

    elif intent == "explain":
        question_text = parsed.get("question_text") or user_input
        with st.chat_message("assistant"):
            full = ""
            placeholder = st.empty()
            for token in stream_explanation(question_text):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.history.append({"role": "assistant", "content": full})

    else:
        with st.chat_message("assistant"):
            full = ""
            placeholder = st.empty()
            for token in stream_general(user_input):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.history.append({"role": "assistant", "content": full})