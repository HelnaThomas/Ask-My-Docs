import json
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.ingest import CHROMA_DIR, CHUNKS_CACHE, COLLECTION_NAME

class HybridRerankedRetriever:
    def __init__(self, reranker_model: str = "BAAI/bge-reranker-base"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        self.chunks: List[Document] = self._load_chunks()
        self.id_to_idx: Dict[str, int] = {
            c.metadata["chunk_id"]: i for i, c in enumerate(self.chunks)
        }
        tokenized = [c.page_content.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.reranker = CrossEncoder(reranker_model)

    def _load_chunks(self) -> List[Document]:
        docs = []
        with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
        return docs

    def _reciprocal_rank_fusion(
        self, bm25_ids: List[str], vector_ids: List[str], k: int = 60
    ) -> List[str]:
        rrf_scores: Dict[str, float] = {}
        for rank, cid in enumerate(bm25_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, cid in enumerate(vector_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    def retrieve(
        self, query: str, candidate_k: int = 25, top_n: int = 5, min_score: float = 0.25
    ) -> List[Tuple[Document, float]]:
        # 1. BM25 lexical search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_ids = [self.chunks[i].metadata["chunk_id"] for i in top_bm25_indices]

        # 2. Vector semantic search
        vec_docs = self.vectorstore.similarity_search(query, k=candidate_k)
        vector_ids = [d.metadata["chunk_id"] for d in vec_docs]

        # 3. Hybrid fusion
        fused_ids = self._reciprocal_rank_fusion(bm25_ids, vector_ids)[:candidate_k]
        candidates = [self.chunks[self.id_to_idx[cid]] for cid in fused_ids if cid in self.id_to_idx]

        if not candidates:
            return []

        # 4. Cross-encoder pair scoring
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x, reverse=True)

        return [(doc, score) for doc, score in scored_docs[:top_n] if score >= min_score]
