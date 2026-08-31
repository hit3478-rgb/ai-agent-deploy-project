"""
RAGAS 평가 스크립트

Agent Workflow(rag_pipeline + agent_workflow)를 테스트 질문 세트로 돌려서
question / answer / contexts를 수집한 뒤 평가하고 CSV로 저장한다.

세 가지 모드, 아래 우선순위로 자동 선택된다.

1) Ollama가 떠있으면 → 진짜 RAGAS, Ollama(llama3.3)를 판정관으로 사용 (기본/추천)
   API 키 필요 없음 — 이미 쓰는 로컬 모델을 그대로 재사용.
   실행 (RAGAS 전용 격리 가상환경 필요 — 메인 앱과 의존성 충돌 방지):
       python3 -m venv ragas_env
       ./ragas_env/bin/pip install -r requirements-ragas.txt
       ./ragas_env/bin/python evaluate_ragas.py

2) Ollama는 없는데 ANTHROPIC_API_KEY가 있을 때 → 진짜 RAGAS, Claude를 판정관으로 사용

3) 둘 다 없을 때 → 경량 대체 지표 (TF-IDF 코사인 유사도 기반, LLM 불필요)
   faithfulness_proxy / answer_relevancy_proxy / context_precision_proxy
   실행 (메인 앱 환경 그대로, 추가 설치 불필요): python3 evaluate_ragas.py

평가 지표 (진짜 RAGAS 모드)
- faithfulness      : 답변이 참고 문서(context) 내용에서 벗어나지 않았는지 (환각 여부)
- answer_relevancy  : 답변이 질문 의도에 실제로 부합하는지 (임베딩 필요 — mxbai-embed-large 재사용)
- context_precision : 검색된 문서들이 질문과 얼마나 관련 있는지 (RAG 검색 품질 자체 평가)
"""

import os
import sys

from rag_pipeline import load_documents, chunk_documents, AdvancedRAGIndex
from agent_workflow import build_workflow, run_agent

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11440")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

EVAL_QUERIES = [
    "패딩 재고 있나요?",
    "반품하려면 어떻게 해야 하나요?",
]

# RAGAS의 context_precision/context_recall은 "정답(reference)"이 있어야 채점 가능하다.
# 우리 데이터(rag_product_data.txt)에 실제로 적힌 사실을 근거로 정답을 직접 작성했다.
REFERENCE_ANSWERS = {
    "패딩 재고 있나요?": "프리미엄 구스다운 롱패딩은 각 사이즈 재고가 있다.",
    "반품하려면 어떻게 해야 하나요?": "단순 변심 반품은 상품 수령 후 7일 이내 가능하며 왕복 배송비는 고객이 부담한다.",
    "운동화 사이즈 종류 알려줘": "데일리 러닝화 에어쿠션은 230~280 사이즈가 있으며 260 사이즈는 품절이다.",
    "배송은 얼마나 걸려요?": "일반 상품은 결제 완료 후 평균 2~3일 이내 배송되며, 당일출고 상품은 오후 2시 이전 주문 시 익일 도착한다.",
    "패딩 소재가 뭐예요?": "구스다운 90%, 페더 10% 소재이다.",
    "무료배송 기준이 뭔가요?": "3만원 이상 구매 시 배송비가 무료다.",
}


def is_ollama_available() -> bool:
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def collect_eval_data():
    raw = load_documents("product_data/rag_product_data.txt")
    chunks = chunk_documents(raw)
    index = AdvancedRAGIndex(chunks)
    workflow = build_workflow(index)

    rows = []
    for q in EVAL_QUERIES:
        result = run_agent(workflow, q)
        contexts = [index.id_to_chunk[sid]["text"] for sid in result["sources"] if sid in index.id_to_chunk]
        if not contexts:
            contexts = [""]
        rows.append({"question": q, "answer": result["answer"], "contexts": contexts, "reference": REFERENCE_ANSWERS[q]})
    return rows


# ---------- 모드 3: LLM 없이 돌아가는 경량 대체 지표 ----------
def compute_proxy_metrics(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    results = []
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        context_text = " ".join(row["contexts"])

        corpus = [question, answer, context_text]
        vectorizer = TfidfVectorizer()
        try:
            matrix = vectorizer.fit_transform(corpus)
            sim = cosine_similarity(matrix)
            answer_relevancy = float(sim[0][1])
            faithfulness = float(sim[1][2])
            context_precision = float(sim[0][2])
        except ValueError:
            answer_relevancy = faithfulness = context_precision = 0.0

        results.append({
            "question": question,
            "answer": answer[:200],
            "faithfulness_proxy": round(faithfulness, 4),
            "answer_relevancy_proxy": round(answer_relevancy, 4),
            "context_precision_proxy": round(context_precision, 4),
        })
    return results


def run_proxy_mode():
    import csv

    print("[안내] Ollama/ANTHROPIC_API_KEY 둘 다 없어 경량 대체 지표(TF-IDF 기반)로 평가합니다.\n")

    print("평가 데이터 수집 중 (Agent Workflow 실행)...")
    rows = collect_eval_data()
    for r in rows:
        print(f"  - {r['question']}  (참조 {len(r['contexts'])}개)")

    print("\n프록시 지표 계산 중...")
    scored = compute_proxy_metrics(rows)

    out_path = "ragas_eval_result_proxy.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    print(f"\n평가 완료. 결과 저장: {out_path}")
    print(f"{'질문':<25}{'faithfulness':>14}{'relevancy':>12}{'precision':>12}")
    for r in scored:
        print(f"{r['question']:<25}{r['faithfulness_proxy']:>14}{r['answer_relevancy_proxy']:>12}{r['context_precision_proxy']:>12}")


# ---------- 모드 1 & 2: 진짜 RAGAS (LLM 채점) ----------
def run_real_ragas(judge_llm, judge_embeddings, judge_name: str):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    print(f"[안내] 판정관: {judge_name}\n")
    print("평가 데이터 수집 중 (Agent Workflow 실행)...")
    rows = collect_eval_data()
    for r in rows:
        print(f"  - {r['question']}  (참조 {len(r['contexts'])}개)")

    dataset = Dataset.from_list(rows)

    from ragas.run_config import RunConfig
    # llama3.3처럼 무거운 로컬 모델은 병렬 요청을 감당 못 해 타임아웃 나기 쉽다.
    # 동시 요청 수를 1로 낮춰 순차 처리하고, 요청당 타임아웃도 넉넉히 늘린다.
    run_config = RunConfig(max_workers=1, timeout=600)

    print("\nRAGAS 평가 실행 중 (LLM 채점 — 순차 처리라 다소 오래 걸립니다, 질문당 수십 초~수 분)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    out_path = "ragas_eval_result.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n평가 완료. 결과 저장: {out_path}")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]])


def main():
    if is_ollama_available():
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        judge_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        judge_embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        run_real_ragas(judge_llm, judge_embeddings, f"Ollama {OLLAMA_MODEL} (로컬, 무료)")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        from langchain_community.embeddings import HuggingFaceEmbeddings
        judge_llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)
        # Anthropic은 임베딩 API가 없어 answer_relevancy용으로 별도 임베딩 필요
        judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        run_real_ragas(judge_llm, judge_embeddings, "Claude (claude-sonnet-5)")
    else:
        run_proxy_mode()


if __name__ == "__main__":
    main()
