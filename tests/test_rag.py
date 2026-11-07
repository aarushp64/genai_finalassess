"""
Unit Tests for RAG and Document Processing Engine
Author: Aarush Parate (Roll No: 1)
"""

import unittest
from modules.document_processor import DocumentProcessor
from modules.rag_engine import RAGEngine


class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        self.rag = RAGEngine()

    def test_text_chunking(self):
        sample_text = (
            "Neural networks learn representations from data. "
            "Deep learning stacks multiple layers to extract hierarchical features. "
            "Transformers use attention mechanisms to model long-range context without recurrence."
        )
        chunks = DocumentProcessor.chunk_text(sample_text, "test_doc.txt", chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "test_doc.txt")

    def test_faiss_indexing_and_retrieval(self):
        chunks = [
            {"chunk_id": 0, "source": "notes.txt", "text": "Convolutional networks are used for image processing."},
            {"chunk_id": 1, "source": "notes.txt", "text": "Self-attention mechanisms are the foundation of modern transformers."},
            {"chunk_id": 2, "source": "notes.txt", "text": "Backpropagation computes gradients using the chain rule."},
        ]

        count = self.rag.index_chunks(chunks)
        self.assertEqual(count, 3)

        # Retrieve for transformer query
        results = self.rag.retrieve("How do transformers work with attention?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Self-attention", results[0]["text"])
        self.assertGreater(results[0]["similarity_score"], 0.4)


if __name__ == "__main__":
    unittest.main()
