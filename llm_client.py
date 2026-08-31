"""
LLM 연동 모듈 — 검색된 문서(context)를 근거로 실제 자연어 답변을 생성.

우선순위:
1. Ollama (H200에 띄워둔 오픈소스 모델, 기본 llama3.3) — 비용 없음, 기본값
2. Anthropic API (ANTHROPIC_API_KEY 있을 때) — 선택사항
3. 둘 다 안 되면 None 반환 → 호출부가 검색 결과 원문 노출로 폴백

환경변수:
    OLLAMA_BASE_URL   (기본값: http://localhost:11440)
    OLLAMA_MODEL      (기본값: llama3.3)
    ANTHROPIC_API_KEY (선택)
"""

import os

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11440")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.3")
ANTHROPIC_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "당신은 쇼핑몰 상품 상담 AI Agent입니다. "
    "아래 제공된 참고 문서만 근거로 답변하세요. "
    "문서에 없는 내용은 추측하지 말고, 모르면 모른다고 답하세요. "
    "친절하고 간결한 한국어로, 2~4문장 이내로 답하세요."
)


def is_ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def is_anthropic_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _generate_via_ollama(query: str, contexts: list[str]) -> str | None:
    try:
        context_text = "\n\n---\n\n".join(contexts)
        user_msg = f"[참고 문서]\n{context_text}\n\n[질문]\n{query}"

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except Exception as e:
        print(f"[llm_client] Ollama 호출 실패, 폴백 처리: {e}")
        return None


def _generate_via_anthropic(query: str, contexts: list[str]) -> str | None:
    try:
        import anthropic

        client = anthropic.Anthropic()
        context_text = "\n\n---\n\n".join(contexts)
        user_msg = f"[참고 문서]\n{context_text}\n\n[질문]\n{query}"

        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    except Exception as e:
        print(f"[llm_client] Anthropic 호출 실패, 폴백 처리: {e}")
        return None


def generate_answer(query: str, contexts: list[str]) -> str | None:
    """Ollama 우선, 안 되면 Anthropic, 둘 다 안 되면 None (호출부가 폴백 처리)."""
    if is_ollama_available():
        answer = _generate_via_ollama(query, contexts)
        if answer:
            return answer

    if is_anthropic_available():
        answer = _generate_via_anthropic(query, contexts)
        if answer:
            return answer

    return None
