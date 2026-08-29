"""
RAGAS 평가 스크립트

Agent Workflow(rag_pipeline + agent_workflow)를 테스트 질문 세트로 돌려서
question / answer / contexts를 수집한 뒤 평가하고 CSV로 저장한다.

두 가지 모드로 동작한다.

1) ANTHROPIC_API_KEY가 있을 때 → 진짜 RAGAS (LLM이 채점)
   faithfulness / answer_relevancy / context_precision
   실행 (RAGAS 전용 격리 가상환경 필요 — 메인 앱과 의존성 충돌 방지):
       python3 -m venv ragas_env
       ./ragas_env/bin/pip install -r requirements-ragas.txt
       ANTHROPIC_API_KEY="sk-ant-..." ./ragas_env/bin/python evaluate_ragas.py

2) ANTHROPIC_API_KEY가 없을 때 → 경량 대체 지표 (TF-IDF 코사인 유사도 기반, LLM 불필요)
   faithfulness_proxy / answer_relevancy_proxy / context_precision_proxy
   실행 (메인 앱 환경 그대로, 추가 설치 불필요):
       python3 evaluate_ragas.py

   주의: 이 프록시 지표는 "의미"가 아니라 "표면적 단어 겹침"만 보기 때문에
   진짜 RAGAS(LLM이 의미를 읽고 판단)보다 정밀도가 낮다. 급하게 지금 제출할 CSV가
   필요할 때 쓰는 임시 대체 수단이고, 키가 생기면 1)번으로 전환해서 다시 돌리는 게 맞다.
"""

import os
import sys

from rag_pipeline import load_documents, chunk_documents, AdvancedRAGIndex
from agent_workflow import build_workflow, run_agent

EVAL_QUERIES = [
    "패딩 재고 있나요?",
    "반품하려면 어떻게 해야 하나요?",
    "운동화 사이즈 종류 알려줘",
    "배송은 얼마나 걸려요?",
    "패딩 소재가 뭐예요?",
    "무료배송 기준이 뭔가요?",
]


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
        rows.append({"question": q, "answer": result["answer"], "contexts": contexts})
    return rows


# ---------- 모드 2: LLM 없이 돌아가는 경량 대체 지표 ----------
def compute_proxy_metrics(rows):
    """
    RAGAS의 세 지표를 흉내낸 TF-IDF 코사인 유사도 기반 프록시.
    - faithfulness_proxy      : 답변이 context와 얼마나 겹치는가 (환각 여부의 근사치)
    - answer_relevancy_proxy  : 답변이 질문과 얼마나 겹치는가
    - context_precision_proxy : context가 질문과 얼마나 겹치는가 (검색 품질 근사치)
    """
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
            # sim[0]=question, sim[1]=answer, sim[2]=context 간 유사도 행렬
            answer_relevancy = float(sim[0][1])
            faithfulness = float(sim[1][2])
            context_precision = float(sim[0][2])
        except ValueError:
            # 어휘가 하나도 안 겹치는 극단적 케이스
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

    print("[안내] API_KEY가 없어 경량 대체 지표(TF-IDF 기반)로 평가합니다.")
    print("       진짜 RAGAS가 필요하면 키 설정 후 다시 실행하세요.\n")

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


# ---------- 모드 1: 진짜 RAGAS (LLM 채점) ----------
def run_real_ragas():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_anthropic import ChatAnthropic

    print("평가 데이터 수집 중 (Agent Workflow 실행)...")
    rows = collect_eval_data()
    for r in rows:
        print(f"  - {r['question']}  (참조 {len(r['contexts'])}개)")

    dataset = Dataset.from_list(rows)
    judge_llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)

    print("\nRAGAS 평가 실행 중 (LLM 채점 — 다소 시간이 걸립니다)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
    )

    df = result.to_pandas()
    out_path = "ragas_eval_result.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n평가 완료. 결과 저장: {out_path}")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]])


def main():
    if os.environ.get("ANTHROPIC_API_KEY"):
        run_real_ragas()
    else:
        run_proxy_mode()


if __name__ == "__main__":
    main()
