"""
ChromaDB + mxbai-embed-large(Ollama 임베딩) 기반 Vector DB

- Embedding Model : mxbai-embed-large (Ollama로 서빙 — 서버에 이미 설치되어 있어 추가 다운로드 불필요)
- Vector DB       : ChromaDB (로컬 영속 저장, ./chroma_db 폴더)
- 저장하는 데이터  : rag_product_data.txt를 청킹한 상품/FAQ/정책 문서 (문서 ID, 원문 텍스트)

실행 (H200에서, Ollama가 떠있어야 함):
    python3 vector_store.py     # DB 빌드 + 검색 테스트
"""

import os
import requests
import chromadb

from rag_pipeline import load_documents, chunk_documents

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11440")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "product_agent_docs"


def embed_text(text: str) -> list[float]:
    """Ollama의 /api/embeddings 엔드포인트로 텍스트를 벡터로 변환."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


import chromadb.utils.embedding_functions as chroma_ef


class OllamaEmbeddingFunction(chroma_ef.EmbeddingFunction):
    """ChromaDB 최신(1.x) EmbeddingFunction 프로토콜 구현체."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [embed_text(t) for t in input]

    def name(self) -> str:
        return f"ollama-{EMBEDDING_MODEL}"

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction()

    def get_config(self) -> dict:
        return {}


def build_vector_db(source_path: str = "product_data/rag_product_data.txt") -> chromadb.api.models.Collection.Collection:
    """rag_product_data.txt를 읽어 청킹 -> mxbai-embed-large 임베딩 -> ChromaDB 저장."""
    raw = load_documents(source_path)
    chunks = chunk_documents(raw)

    print(f"Loading documents...")
    print(f"TXT documents: 1 (내부에 {len(chunks)}개 청크)")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # 기존 컬렉션 있으면 삭제 후 재생성 (재빌드 시 중복 방지)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
    )

    print("Chunking completed")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Saving ChromaDB...")

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": c["id"]} for c in chunks],
    )

    print("Vector DB Build Complete")
    print(f"저장 위치: {CHROMA_DIR}")
    return collection


class VectorRAGIndex:
    """agent_workflow.py의 rag_index 인터페이스(.search(query, top_k) -> list[dict])와 호환되는 벡터 검색 인덱스."""

    def __init__(self, chroma_dir: str = CHROMA_DIR):
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddingFunction(),
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        out = []
        for doc_id, text, distance in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            # ChromaDB는 거리(distance)를 반환 (작을수록 유사) -> 유사도 점수로 변환
            similarity = 1 / (1 + distance)
            out.append({"id": doc_id, "score": round(similarity, 4), "text": text})
        return out


if __name__ == "__main__":
    build_vector_db()

    print("\n=== 검색 테스트 ===")
    index = VectorRAGIndex()
    for q in ["패딩 재고 있나요?", "반품 정책 알려줘", "배송은 얼마나 걸려요?"]:
        results = index.search(q, top_k=2)
        print(f"\n[질문] {q}")
        for r in results:
            print(f"  (유사도 {r['score']}) {r['id']}")
