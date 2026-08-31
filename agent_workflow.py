"""
조건부 Edge를 이용한 Agent Workflow (LangGraph)

흐름:
    START
      └─ classify (질문 종류 분류)
            ├─[smalltalk]──────────────► smalltalk_response ─► END
            └─[product_qa]─► rag_search
                                  ├─[검색 신뢰도 낮음]─► fallback_response ─► END
                                  └─[검색 신뢰도 충분]─► answer_from_search ─► END

- 조건부 Edge #1 : 질문 종류(잡담 vs 상품문의)에 따라 RAG를 탈지 말지 분기
- 조건부 Edge #2 : 검색 결과 점수(신뢰도)에 따라 정상 답변할지 "못 찾음" 안내로 갈지 분기
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from llm_client import generate_answer
from rag_pipeline import compress_context

SMALLTALK_KEYWORDS = ["안녕", "반가워", "고마워", "감사", "잘가", "누구야", "너 뭐야"]
SEARCH_CONFIDENCE_THRESHOLD = 0.12  # relevance(원본 word/char 유사도 최댓값) 기준 임계값


class AgentState(TypedDict):
    query: str
    query_type: str          # "smalltalk" | "product_qa"
    search_results: list
    answer: str
    sources: list


def build_workflow(rag_index):
    """rag_index: AdvancedRAGIndex 인스턴스를 주입받아 그래프를 구성한다."""

    # ---------- 노드 ----------
    def classify(state: AgentState) -> AgentState:
        query = state["query"]
        is_smalltalk = any(kw in query for kw in SMALLTALK_KEYWORDS)
        state["query_type"] = "smalltalk" if is_smalltalk else "product_qa"
        return state

    def smalltalk_response(state: AgentState) -> AgentState:
        state["answer"] = "안녕하세요! 상품 검색, 가격, 재고, 배송, 반품 등 무엇이든 물어보세요."
        state["sources"] = []
        return state

    def rag_search(state: AgentState) -> AgentState:
        results = rag_index.search(state["query"], top_k=2)
        state["search_results"] = results
        return state

    def answer_from_search(state: AgentState) -> AgentState:
        query = state["query"]
        # Advanced RAG 기법 4: Context Compression — 문서 원문을 통째로 넘기지 않고
        # 질문과 관련된 줄만 추려서 LLM에 전달 (토큰 절약 + 무관한 정보 혼입 방지)
        contexts = [compress_context(query, r["text"]) for r in state["search_results"]]
        llm_answer = generate_answer(query, contexts)
        if llm_answer:
            state["answer"] = llm_answer
        else:
            # LLM 미연동/실패 시 폴백: 압축된 컨텍스트를 그대로 노출
            state["answer"] = f"[{contexts[0].splitlines()[0]}] 관련 정보를 찾았습니다.\n\n{contexts[0]}"
        state["sources"] = [r["id"] for r in state["search_results"]]
        return state

    def fallback_response(state: AgentState) -> AgentState:
        state["answer"] = "관련된 상품/정책 정보를 찾지 못했습니다. 다른 표현으로 질문해주세요."
        state["sources"] = [r["id"] for r in state.get("search_results", [])]
        return state

    # ---------- 조건부 Edge 분기 함수 ----------
    def route_after_classify(state: AgentState) -> str:
        return "smalltalk" if state["query_type"] == "smalltalk" else "rag_search"

    def route_after_search(state: AgentState) -> str:
        results = state["search_results"]
        if not results or results[0]["relevance"] < SEARCH_CONFIDENCE_THRESHOLD:
            return "fallback"
        return "confident"

    # ---------- 그래프 구성 ----------
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("smalltalk_response", smalltalk_response)
    graph.add_node("rag_search", rag_search)
    graph.add_node("answer_from_search", answer_from_search)
    graph.add_node("fallback_response", fallback_response)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"smalltalk": "smalltalk_response", "rag_search": "rag_search"},
    )
    graph.add_conditional_edges(
        "rag_search",
        route_after_search,
        {"confident": "answer_from_search", "fallback": "fallback_response"},
    )

    graph.add_edge("smalltalk_response", END)
    graph.add_edge("answer_from_search", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


def run_agent(compiled_graph, query: str) -> dict:
    result = compiled_graph.invoke({"query": query, "query_type": "", "search_results": [], "answer": "", "sources": []})
    return {"answer": result["answer"], "sources": result["sources"], "query_type": result["query_type"]}
