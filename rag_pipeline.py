"""
기본 RAG 파이프라인 데모
1. 문서 로드 → 2. 청킹 → 3. 임베딩(TF-IDF) → 4. 인덱싱 → 5. 검색(Retrieval)

실제 서비스 전환 시 교체 지점:
- 3번 임베딩: TfidfVectorizer → OpenAI text-embedding-3-small / Cohere embed 등
- 4번 인덱싱: in-memory matrix → Pinecone / Chroma / Weaviate 등 벡터DB
- 검색 이후: 검색된 chunk를 LLM 프롬프트에 넣어 최종 답변 생성 (Generation 단계, 별도 구현 필요)
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 1. 문서 로드 ----------
def load_documents(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- 2. 청킹 ----------
def chunk_documents(raw_text: str) -> list[dict]:
    """
    '[문서 ID: XXX]' 블록 단위로 분할.
    실제 데이터에서는 문서 길이가 길면 500~1000자 단위 슬라이딩 윈도우 청킹도 병행.
    """
    blocks = re.split(r"\n\n(?=\[문서 ID:)", raw_text.strip())
    chunks = []
    for block in blocks:
        match = re.search(r"\[문서 ID:\s*(.+?)\]", block)
        doc_id = match.group(1).strip() if match else "UNKNOWN"
        chunks.append({"id": doc_id, "text": block.strip()})
    return chunks


# ---------- 3. 임베딩 + 4. 인덱싱 ----------
class SimpleVectorIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        corpus = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)  # 인덱싱된 벡터 저장소

    # ---------- 5. 검색 ----------
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked:
            results.append({
                "id": self.chunks[idx]["id"],
                "score": round(float(scores[idx]), 4),
                "text": self.chunks[idx]["text"],
            })
        return results


# ---------- 실행 ----------
if __name__ == "__main__":
    raw = load_documents("product_data/rag_product_data.txt")  # 로컬 실행 시 이 경로 사용
    chunks = chunk_documents(raw)
    print(f"총 {len(chunks)}개 청크 로드 완료\n")

    index = SimpleVectorIndex(chunks)

    test_queries = [
        "패딩 재고 있나요?",
        "반품하려면 어떻게 해야 하나요?",
        "운동화 사이즈 종류 알려줘",
        "배송은 얼마나 걸려요?",
        "패딩 소재가 뭐예요?",
    ]

    for q in test_queries:
        print(f"\n[질문] {q}")
        results = index.search(q, top_k=2)
        for r in results:
            print(f"  - (유사도 {r['score']}) [{r['id']}]")
            preview = r["text"].split("\n")[1] if "\n" in r["text"] else r["text"][:40]
            print(f"    {preview}")