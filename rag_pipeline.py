"""
RAG 파이프라인 - Advanced RAG 버전
1. 문서 로드 → 2. 청킹 → 3. 임베딩(단어 TF-IDF + 문자 n-gram TF-IDF)
→ 4. 인덱싱 → 5. Multi-Query 검색 → 6. Reranking

Advanced RAG 적용 내역
- Multi-Query: 질문 하나를 도메인 동의어 사전으로 2~3개 변형 질의로 확장해 각각 검색 후
  RRF(Reciprocal Rank Fusion)로 결과를 병합. 사용자가 안 쓴 단어라서 놓치는 케이스를 줄임.
- Reranking: 1차 후보(word-level TF-IDF)를 문자 n-gram TF-IDF로 재채점해 최종 순위 결정.
  한국어는 조사가 단어 끝에 붙는 교착어라 word-level 토큰화가 취약한데,
  문자 단위 n-gram은 '소재가'와 '소재:'처럼 표기가 달라도 부분 문자열이 겹치면 유사도를 잡아낸다.

실제 서비스 전환 시 교체 지점
- 임베딩: TF-IDF → OpenAI/Cohere 임베딩, 문자 n-gram → 실제 Cross-Encoder Reranker
- Multi-Query 생성: 규칙 기반 동의어 사전 → LLM에게 "질문을 3가지로 바꿔써줘" 요청
- 인덱싱: in-memory matrix → Pinecone / Chroma / Weaviate
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
    blocks = re.split(r"\n\n(?=\[문서 ID:)", raw_text.strip())
    chunks = []
    for block in blocks:
        match = re.search(r"\[문서 ID:\s*(.+?)\]", block)
        doc_id = match.group(1).strip() if match else "UNKNOWN"
        chunks.append({"id": doc_id, "text": block.strip()})
    return chunks


# ---------- 3. 기존 버전 (하위 호환용으로 유지) ----------
class SimpleVectorIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        corpus = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"id": self.chunks[i]["id"], "score": round(float(scores[i]), 4), "text": self.chunks[i]["text"]}
            for i in ranked
        ]


# ---------- Multi-Query 확장용 도메인 동의어 사전 ----------
SYNONYM_MAP = {
    "재고": ["품절", "수량", "남은"],
    "가격": ["얼마", "금액", "비용"],
    "배송": ["택배", "도착", "소요"],
    "반품": ["환불", "교환"],
    "소재": ["원단", "성분", "재질"],
    "사이즈": ["치수", "크기", "옵션"],
    "색상": ["컬러", "색깔"],
}


def expand_queries(query: str, max_variants: int = 2) -> list[str]:
    """규칙 기반 Multi-Query 생성: 동의어 사전에 매칭되는 키워드를 치환해 변형 질의를 만든다."""
    variants = [query]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query:
            for syn in synonyms[:max_variants]:
                variants.append(query.replace(key, syn))
    # 중복 제거, 원 질의 포함 최대 3개까지만
    seen, unique = set(), []
    for v in variants:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique[:3]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict:
    """여러 검색 결과(문서 ID 순위 리스트)를 RRF 공식으로 병합해 점수를 매긴다."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ---------- Advanced RAG 인덱스 (Multi-Query + Reranking) ----------
class AdvancedRAGIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.id_to_chunk = {c["id"]: c for c in chunks}
        corpus = [c["text"] for c in chunks]

        # 1차 검색용: 단어 단위 TF-IDF
        self.word_vectorizer = TfidfVectorizer()
        self.word_matrix = self.word_vectorizer.fit_transform(corpus)

        # 재정렬(rerank)용: 문자 n-gram TF-IDF (한국어 조사 변형에 더 강함)
        self.char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)

    def _word_scores(self, query: str) -> dict:
        """모든 문서에 대한 word-level 원본 코사인 유사도 (RRF 순위 왜곡과 무관한 진짜 점수)"""
        query_vec = self.word_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.word_matrix).flatten()
        return {self.chunks[i]["id"]: float(scores[i]) for i in range(len(self.chunks))}

    def _word_search_ids(self, query: str, top_k: int) -> list[str]:
        query_vec = self.word_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.word_matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [self.chunks[i]["id"] for i in ranked]

    def _char_search_ids(self, query: str, top_k: int) -> list[str]:
        query_vec = self.char_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.char_matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [self.chunks[i]["id"] for i in ranked]

    def _char_score(self, query: str, doc_id: str) -> float:
        idx = next(i for i, c in enumerate(self.chunks) if c["id"] == doc_id)
        query_vec = self.char_vectorizer.transform([query])
        score = cosine_similarity(query_vec, self.char_matrix[idx]).flatten()[0]
        return float(score)

    def search(self, query: str, top_k: int = 3, candidate_k: int = 6) -> list[dict]:
        # 1) Multi-Query 확장
        queries = expand_queries(query)

        # 2) 후보 생성: 질의별 word-level 검색 + 원 질의의 char n-gram 검색을 함께 RRF로 병합
        ranked_lists = [self._word_search_ids(q, candidate_k) for q in queries]
        ranked_lists.append(self._char_search_ids(query, candidate_k))
        fused_scores = reciprocal_rank_fusion(ranked_lists)
        candidates = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)[:candidate_k]

        if not candidates:
            return []

        # 3) Reranking + 진짜 관련도(relevance) 계산
        #    relevance는 RRF 순위가 아니라 원본 코사인 유사도(word/char) 중 최댓값 —
        #    "후보 목록에 들어만 있어도 점수가 생기는" RRF의 왜곡을 걸러내는 신뢰도 게이트로 사용
        max_fused = max(fused_scores.values()) or 1.0
        raw_word_scores = self._word_scores(query)

        reranked = []
        for doc_id in candidates:
            rrf_norm = fused_scores[doc_id] / max_fused
            char_sim = self._char_score(query, doc_id)
            word_sim = raw_word_scores.get(doc_id, 0.0)
            final_score = rrf_norm * 0.4 + char_sim * 0.6
            relevance = max(word_sim, char_sim)  # 진짜 내용 일치도
            reranked.append((doc_id, final_score, char_sim, relevance))

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked = reranked[:top_k]

        return [
            {
                "id": doc_id,
                "score": round(final_score, 4),
                "char_similarity": round(char_sim, 4),
                "relevance": round(relevance, 4),
                "text": self.id_to_chunk[doc_id]["text"],
            }
            for doc_id, final_score, char_sim, relevance in reranked
        ]


# ---------- 실행 (직접 실행 시 word-only vs Advanced 비교) ----------
if __name__ == "__main__":
    raw = load_documents("product_data/rag_product_data.txt")
    chunks = chunk_documents(raw)
    print(f"총 {len(chunks)}개 청크 로드 완료\n")

    simple_index = SimpleVectorIndex(chunks)
    advanced_index = AdvancedRAGIndex(chunks)

    test_queries = [
        "패딩 재고 있나요?",
        "반품하려면 어떻게 해야 하나요?",
        "운동화 사이즈 종류 알려줘",
        "배송은 얼마나 걸려요?",
        "패딩 소재가 뭐예요?",
    ]

    for q in test_queries:
        print(f"\n[질문] {q}")
        print(f"  질의 확장: {expand_queries(q)}")

        print("  -- 기존(TF-IDF only) --")
        for r in simple_index.search(q, top_k=2):
            print(f"     (score {r['score']}) {r['id']}")

        print("  -- Advanced RAG (Multi-Query + Rerank) --")
        for r in advanced_index.search(q, top_k=2):
            print(f"     (score {r['score']}, char_sim {r['char_similarity']}) {r['id']}")
