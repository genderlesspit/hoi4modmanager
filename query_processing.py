#!/usr/bin/env python3
"""
Query Processing Module
========================
Handles chunk ranking, similarity searches, and token-limited accumulation.
"""

import logging
import numpy as np
from hoi4_vector_db import HOI4VectorDB


class ChunkRanker:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def rank_chunks(self, query, top_k=5, similarity_threshold=0.75):
        query_embedding = np.array(self.vector_db.embed_text(query)).astype(np.float32)
        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k, where={"chunk_type": "standard"}, include=["embeddings", "documents"]
        )
        if not results.get("documents") or not results["documents"][0]:
            logging.warning("No standard chunks found for the query.")
            return []
        docs = results["documents"][0]
        embeddings = np.array(results["embeddings"][0]).astype(np.float32)
        similarities = np.dot(query_embedding, embeddings.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(embeddings, axis=1))
        ranked = [(None, doc, sim) for doc, sim in zip(docs, similarities) if sim >= similarity_threshold]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked


class ContextChunkRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def get_context_chunk(self, standard_chunk_text):
        query_embedding = np.array(self.vector_db.embed_text(standard_chunk_text)).astype(np.float32)
        results = self.vector_db.collection.query(query_embeddings=[query_embedding.tolist()], n_results=1, where={"chunk_type": "context"}, include=["documents"])
        return results["documents"][0][0] if results.get("documents") and results["documents"][0] else ""


class TokenLimitManager:
    def __init__(self, max_tokens=1500):
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(text.split())

    def accumulate_context(self, context_chunks):
        accumulated = ""
        for chunk in context_chunks:
            if self.count_tokens(accumulated + "\n" + chunk) <= self.max_tokens:
                accumulated += "\n" + chunk
            else:
                break
        return accumulated.strip()