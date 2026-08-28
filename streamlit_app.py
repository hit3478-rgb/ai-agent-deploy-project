"""
쇼핑몰 상품 상담 AI Agent — Streamlit 프론트엔드

실행:
    streamlit run streamlit_app.py --server.port 8501
"""

import os
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8010")

st.set_page_config(page_title="상품 상담 AI Agent", page_icon="🛍️")
st.title("🛍️ 쇼핑몰 상품 상담 AI Agent")
st.caption(f"백엔드: {API_BASE}")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

# ---------- 지난 대화 렌더링 ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- 입력창 ----------
query = st.chat_input("예) 패딩 재고 있나요?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            with requests.get(
                f"{API_BASE}/chat/stream",
                params={
                    "query": query,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=30,
            ) as resp:
                event_type = None
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None or raw_line == "":
                        continue
                    if raw_line.startswith("event:"):
                        event_type = raw_line[len("event:"):]
                        if event_type.startswith(" "):
                            event_type = event_type[1:]
                    elif raw_line.startswith("data:"):
                        data = raw_line[len("data:"):]
                        if data.startswith(" "):
                            data = data[1:]
                        if event_type == "session":
                            st.session_state.session_id = data
                        elif event_type == "token":
                            full_answer += "\n" if data == "\\n" else data
                            placeholder.markdown(full_answer + "▌")
                        elif event_type == "done":
                            placeholder.markdown(full_answer)
                            st.caption(f"참조 문서: {data}")
        except requests.exceptions.RequestException as e:
            full_answer = f"백엔드 연결 실패: {e}"
            placeholder.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})

with st.sidebar:
    st.subheader("세션 정보")
    st.code(st.session_state.session_id or "(아직 없음)")
    if st.button("대화 초기화"):
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
            except requests.exceptions.RequestException:
                pass
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
