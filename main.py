"""
쇼핑몰 상품 상담 AI Agent — 백엔드 서버
기획서 End Point 정의를 구현한 FastAPI 서버.

실행:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger 문서:
    http://<서버주소>:8000/docs
"""

import asyncio
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag_pipeline import load_documents, chunk_documents, AdvancedRAGIndex
from agent_workflow import build_workflow, run_agent

app = FastAPI(
    title="쇼핑몰 상품 상담 AI Agent API",
    description=(
        "캡스톤 프로젝트 - RAG 기반 상품 상담 Agent.\n\n"
        "**session_id 안내**: 대화를 이어가려면 첫 응답에서 받은 session_id를 "
        "다음 요청에도 계속 실어 보내야 합니다. 안 보내면 매번 새 대화로 처리됩니다."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 서버 시작 시 RAG 인덱스 1회 구축 ----------
raw = load_documents("product_data/rag_product_data.txt")
CHUNKS = chunk_documents(raw)
INDEX = AdvancedRAGIndex(CHUNKS)
WORKFLOW = build_workflow(INDEX)

# ---------- 세션 State (인메모리 데모용 — 서버 재시작하면 초기화됨) ----------
SESSIONS: dict[str, list[dict]] = {}


def get_or_create_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    SESSIONS.setdefault(sid, [])
    return sid


def append_turn(session_id: str, role: str, text: str):
    SESSIONS[session_id].append(
        {"role": role, "text": text, "ts": datetime.now().isoformat()}
    )


def build_answer(query: str) -> tuple[str, list[str]]:
    """조건부 Edge Agent Workflow(LangGraph)를 통해 답변 생성.
    질문 종류(잡담/상품문의)와 검색 신뢰도에 따라 내부적으로 다른 경로를 탄다."""
    result = run_agent(WORKFLOW, query)
    return result["answer"], result["sources"]


# ---------- 요청/응답 스키마 ----------
class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="이전 대화의 session_id. 처음 대화라면 비워두세요(null) — 서버가 새로 발급합니다.",
        examples=["8c5980ea-c428-40e4-9fa4-cf89f951aaa7"],
    )
    query: str = Field(..., description="사용자 질문", examples=["패딩 재고 있나요?"])


class ChatResponse(BaseModel):
    session_id: str = Field(description="다음 요청에 그대로 실어 보내야 하는 대화 식별자")
    query: str
    answer: str
    sources: list[str] = Field(description="답변 근거가 된 문서 ID 목록")


# ---------- POST /chat ----------
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="일반 채팅 (한 번에 전체 답변 반환)",
)
def chat(req: ChatRequest):
    """
    **파라미터 (Body / JSON)**
    - `query` (필수): 사용자 질문
    - `session_id` (선택): 이어서 대화할 때만 이전 응답의 session_id를 넣기

    **예시 요청**
    ```json
    { "query": "패딩 재고 있나요?" }
    ```
    두 번째 대화부터:
    ```json
    { "session_id": "8c5980ea-...", "query": "그럼 사이즈는 뭐 있어?" }
    ```
    """
    session_id = get_or_create_session(req.session_id)
    answer, sources = build_answer(req.query)
    append_turn(session_id, "user", req.query)
    append_turn(session_id, "assistant", answer)
    return ChatResponse(session_id=session_id, query=req.query, answer=answer, sources=sources)


# ---------- GET /chat/stream (SSE, 브라우저 EventSource용) ----------
@app.get(
    "/chat/stream",
    tags=["chat"],
    summary="스트리밍 채팅 (SSE, 실시간 토큰 단위 응답)",
)
async def chat_stream(
    query: str = Query(..., description="사용자 질문", examples=["배송은 얼마나 걸려요?"]),
    session_id: str | None = Query(
        default=None,
        description="이전 대화의 session_id. 처음이면 비워두세요.",
    ),
):
    """
    **파라미터 (Query String)** — SSE는 브라우저 EventSource 규격상 GET만 지원하므로
    body가 아닌 쿼리스트링으로 전달합니다.

    - `query` (필수): `/chat/stream?query=배송은 얼마나 걸려요?`
    - `session_id` (선택): `/chat/stream?query=...&session_id=8c5980ea-...`

    **응답 형식 (Server-Sent Events)**
    - `event: session` → 이번 대화의 session_id (최초 1회)
    - `event: token` → 답변 글자를 한 글자씩 순차 전송
    - `event: done` → 스트리밍 종료, data에 참조 문서 ID 목록

    Swagger UI는 SSE 스트림을 실시간으로 보여주지 못하니, 실제 스트리밍 확인은
    curl이나 브라우저 콘솔의 EventSource로 테스트하세요. (아래 안내 참고)
    """
    session_id = get_or_create_session(session_id)
    answer, sources = build_answer(query)
    append_turn(session_id, "user", query)
    append_turn(session_id, "assistant", answer)

    async def event_generator():
        yield {"event": "session", "data": session_id}
        for ch in answer:
            payload = "\\n" if ch == "\n" else ch
            yield {"event": "token", "data": payload}
            await asyncio.sleep(0.01)
        yield {"event": "done", "data": ",".join(sources)}

    return EventSourceResponse(event_generator())


# ---------- GET /products/search ----------
@app.get("/products/search", tags=["products"], summary="키워드 기반 상품/문서 검색")
def search_products(
    keyword: str = Query(..., description="검색어", examples=["운동화"]),
    top_k: int = Query(default=5, description="반환할 결과 개수", ge=1, le=20),
):
    """`/products/search?keyword=운동화&top_k=3` 형태로 호출"""
    results = INDEX.search(keyword, top_k=top_k)
    return {"keyword": keyword, "results": results}


# ---------- GET /products/{id} ----------
@app.get("/products/{product_id}", tags=["products"], summary="상품 상세 조회")
def get_product(product_id: str):
    """path 파라미터로 문서 ID를 그대로 사용 (예: `/products/PRD-002`)"""
    for c in CHUNKS:
        if c["id"] == product_id:
            return {"id": c["id"], "text": c["text"]}
    raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")


# ---------- GET /session/{id}/history ----------
@app.get("/session/{session_id}/history", tags=["session"], summary="대화 히스토리 조회")
def get_history(session_id: str):
    """`/chat` 또는 `/chat/stream` 응답에서 받은 session_id를 path에 그대로 사용"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"session_id": session_id, "history": SESSIONS[session_id]}


# ---------- DELETE /session/{id} ----------
@app.delete("/session/{session_id}", tags=["session"], summary="대화 세션 초기화/삭제")
def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"session_id": session_id, "deleted": True}
    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")


@app.get("/", tags=["health"], summary="헬스체크")
def health():
    return {"status": "ok", "loaded_chunks": len(CHUNKS)}
