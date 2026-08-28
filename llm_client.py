"""
LLM 연동 모듈 — 검색된 문서(context)를 근거로 실제 자연어 답변을 생성.

환경변수 ANTHROPIC_API_KEY가 설정되어 있으면 Claude API를 호출해서 답변을 생성하고,
없으면 None을 반환한다 (호출부에서 기존 방식 - 검색 결과 원문 노출 - 로 자동 폴백).

서버에서 키 설정 방법:
    export ANTHROPIC_API_KEY="sk-ant-..."
    uvicorn main:app --reload
"""

import os

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "당신은 쇼핑몰 상품 상담 AI Agent입니다. "
    "아래 제공된 참고 문서만 근거로 답변하세요. "
    "문서에 없는 내용은 추측하지 말고, 모르면 모른다고 답하세요. "
    "친절하고 간결한 한국어로, 2~4문장 이내로 답하세요."
)


def is_llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_answer(query: str, contexts: list[str]) -> str | None:
    """LLM 기반 답변 생성. 키가 없거나 호출 실패 시 None 반환 (폴백 유도)."""
    if not is_llm_available():
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        context_text = "\n\n---\n\n".join(contexts)
        user_msg = f"[참고 문서]\n{context_text}\n\n[질문]\n{query}"

        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    except Exception as e:
        print(f"[llm_client] LLM 호출 실패, 폴백 처리: {e}")
        return None
