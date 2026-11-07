"""
NeuroTutor AI - RAG & Sentence-BERT Semantic Retrieval Engine
Author: Aarush Parate (Roll No: 1)
Unit III: GPT, BERT and Multimodal Models
Encodes text chunks using Sentence-BERT (all-MiniLM-L6-v2) and performs FAISS similarity search.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from modules.llm_engine import LLMEngine
import config

_EMBEDDER = None


def get_sentence_transformer():
    """Singleton loader for Sentence-BERT embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            _EMBEDDER = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[RAGEngine] Warning loading SentenceTransformer: {e}")
            return None
    return _EMBEDDER


class RAGEngine:
    """Manages dense embedding generation, FAISS indexing, and grounded context synthesis."""

    def __init__(self):
        self.embedder = get_sentence_transformer()
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity on normalized vectors
        self.chunk_store: List[Dict[str, Any]] = []

    def clear(self):
        """Clears FAISS index and chunk store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_store = []

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds chunks using Sentence-BERT and adds normalized vectors to the FAISS index.
        Returns number of newly indexed chunks.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]

        if self.embedder is not None:
            embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)
        else:
            # Fallback simple deterministic embedding
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), self.dimension).astype(np.float32)
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunk_store.extend(chunks)
        return len(chunks)

    def retrieve(
        self, query: str, top_k: int = config.TOP_K_RETRIEVAL, min_score: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k semantically nearest passages for a user query.
        """
        if self.index.ntotal == 0 or not query.strip():
            return []

        if self.embedder is not None:
            query_emb = self.embedder.encode([query], show_progress_bar=False, convert_to_numpy=True)
            query_emb = np.array(query_emb, dtype=np.float32)
            faiss.normalize_L2(query_emb)
        else:
            np.random.seed(len(query))
            query_emb = np.random.randn(1, self.dimension).astype(np.float32)
            faiss.normalize_L2(query_emb)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= min_score and idx < len(self.chunk_store):
                chunk = self.chunk_store[idx].copy()
                chunk["similarity_score"] = round(float(score), 4)
                results.append(chunk)

        return results

    def query_with_grounding(
        self,
        query: str,
        llm: LLMEngine,
        level: str = "Undergraduate Level",
        temperature: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Synthesizes an answer grounded strictly on retrieved academic notes.
        Follows prompt constraints to avoid fabricating unsupported information.
        """
        retrieved_passages = self.retrieve(query, top_k=config.TOP_K_RETRIEVAL)

        # If zero passages pass threshold or index is empty
        if not retrieved_passages:
            return {
                "answer": "I'm not sure based on the available notes. The uploaded documents do not contain sufficient evidence to answer this question accurately.",
                "retrieved_passages": [],
                "is_grounded": False,
                "grounding_status": "Not Supported",
                "grounding_score": 0.0,
            }

        context_str = "\n\n".join(
            [
                f"[Source: {p['source']} | Chunk {p['chunk_id']} | Similarity: {p['similarity_score']:.2f}]\n{p['text']}"
                for p in retrieved_passages
            ]
        )

        rag_prompt = (
            f"You are NeuroTutor AI, a strict syllabus-grounded educational tutor.\n"
            f"Target Audience Level: {level}.\n\n"
            f"STRICT INSTRUCTIONS:\n"
            f"1. You must answer the student's question using ONLY the retrieved facts from the uploaded academic notes below.\n"
            f"2. Clearly cite the source chunk or document when stating facts.\n"
            f"3. Explicitly distinguish retrieved facts from general background context.\n"
            f"4. If the retrieved notes do not contain the answer, you MUST state: \"I'm not sure based on the available notes.\"\n"
            f"5. Do NOT hallucinate or extrapolate beyond what is documented.\n\n"
            f"--- RETRIEVED ACADEMIC NOTES CONTEXT ---\n"
            f"{context_str}\n"
            f"--- END OF CONTEXT ---\n\n"
            f"Student Question: {query}\n\n"
            f"Grounded Explanation:"
        )

        llm_response = llm.generate(
            prompt=rag_prompt,
            temperature=temperature,
            level=level,
        )

        answer_text = llm_response["text"]

        return {
            "answer": answer_text,
            "retrieved_passages": retrieved_passages,
            "latency": llm_response.get("latency", 0.0),
            "provider": llm_response.get("provider", ""),
            "tokens": llm_response.get("tokens_approx", 0),
        }
