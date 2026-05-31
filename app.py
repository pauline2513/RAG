import streamlit as st
from rag_extraction import answer, answer_no_rag


st.set_page_config(layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "pending_mode" not in st.session_state:
    st.session_state.pending_mode = None


col1, col2 = st.columns([1, 4])

with col1:
    mode = st.radio(
        "Выберите режим работы:",
        ("Использовать RAG", "Не использовать RAG")
    )

with col2:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.pending_prompt is not None:
        prompt = st.session_state.pending_prompt
        selected_mode = st.session_state.pending_mode

        with st.chat_message("assistant"):
            with st.spinner("Генерирую ответ..."):
                if selected_mode == "Использовать RAG":
                    response = answer(prompt)
                else:
                    response = answer_no_rag(prompt)

                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.session_state.pending_prompt = None
        st.session_state.pending_mode = None

        st.rerun()

    prompt = st.chat_input(
        "Введите вопрос...",
        disabled=st.session_state.pending_prompt is not None
    )

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        st.session_state.pending_prompt = prompt
        st.session_state.pending_mode = mode

        st.rerun()