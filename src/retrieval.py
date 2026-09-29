"""Vireo Support Intelligence — Hybrid Retrieval Layer.

Provides BM25 + dense vector search with Reciprocal Rank Fusion.
Used for: few-shot retrieval, repeat-contact evidence, representative complaints,
and the "Ask the Tickets" page.

Embeddings are cached to outputs/embeddings.npy keyed by text hash.
No external vector store is used.
"""

import hashlib
import json
import os
import re
import string
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from . import config

# Simple stopwords (keep model-number tokens)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "because", "until", "while", "about", "against", "if", "then",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "this", "that", "these", "those",
    "am", "hi", "hello", "please", "thanks", "thank", "regards",
}


def tokenize(text: str) -> list:
    """Simple tokenisation: lowercase, strip punctuation, remove stopwords,
    but keep model-number tokens (e.g., VA-EB-PL2, VR882745)."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Keep alphanumeric and hyphens (for model numbers)
    tokens = re.findall(r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?', text)
    # Remove stopwords but keep tokens that look like model/order numbers
    result = []
    for t in tokens:
        if t in STOPWORDS and not re.match(r'[a-z]{2}-', t) and not re.match(r'vr\d', t):
            continue
        result.append(t)
    return result


def text_hash(text: str) -> str:
    """SHA-256 hash of text for cache keying."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def reciprocal_rank_fusion(rank_lists: list, k: int = 60) -> dict:
    """Reciprocal Rank Fusion on multiple rank mappings (doc_id -> rank 1-indexed or 0-indexed)."""
    scores = {}
    for r_map in rank_lists:
        for doc_id, rank in r_map.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    return scores


class HybridRetriever:
    """Hybrid BM25 + dense vector retrieval with RRF."""
    
    def __init__(self, texts: list, ids: list = None, embeddings_cache_path: str = None):
        """
        Args:
            texts: List of document texts to index.
            ids: Optional list of IDs corresponding to texts.
            embeddings_cache_path: Path to cache embeddings.
        """
        self.texts = texts
        self.ids = ids or list(range(len(texts)))
        self.cache_path = embeddings_cache_path or os.path.join(
            config.OUTPUTS_DIR, "embeddings.npy"
        )
        self.hash_path = self.cache_path.replace(".npy", "_hashes.json")
        
        # Build BM25 index
        self.tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)
        
        # Dense embeddings (loaded lazily)
        self._embeddings = None
        self._hash_map = {}
    
    def _load_cached_embeddings(self):
        """Load cached embeddings if they exist."""
        if os.path.exists(self.cache_path) and os.path.exists(self.hash_path):
            self._embeddings = np.load(self.cache_path)
            with open(self.hash_path, "r") as f:
                self._hash_map = json.load(f)
            return True
        return False
    
    def _save_embeddings(self, embeddings, hash_map):
        """Save embeddings to cache."""
        np.save(self.cache_path, embeddings)
        with open(self.hash_path, "w") as f:
            json.dump(hash_map, f)
    
    def embed_texts(self, texts_to_embed: list, batch_size: int = 50) -> np.ndarray:
        """Embed texts using Ollama's batch API, with disk caching.

        Sends up to `batch_size` texts per API call (Ollama /api/embed
        accepts a list in the 'input' field).  Cached embeddings are
        reused without calling the API.
        """
        import requests

        hashes = [text_hash(t) for t in texts_to_embed]

        # --- load disk cache ------------------------------------------------
        cached_embeds = {}
        if os.path.exists(self.hash_path):
            with open(self.hash_path, "r") as f:
                self._hash_map = json.load(f)
        if os.path.exists(self.cache_path):
            cached_matrix = np.load(self.cache_path)
            for h, idx in self._hash_map.items():
                if int(idx) < len(cached_matrix):
                    cached_embeds[h] = cached_matrix[int(idx)]

        # --- find what still needs embedding --------------------------------
        to_embed = []
        to_embed_hashes = []
        for i, h in enumerate(hashes):
            if h not in cached_embeds:
                to_embed.append(texts_to_embed[i])
                to_embed_hashes.append(h)

        # --- batch embed via Ollama -----------------------------------------
        new_embeds = {}
        if to_embed:
            n_total = len(to_embed)
            print(f"Embedding {n_total} new texts ({len(cached_embeds)} cached)...")
            for batch_start in range(0, n_total, batch_size):
                batch_texts = [t[:2000] for t in to_embed[batch_start:batch_start + batch_size]]
                batch_h = to_embed_hashes[batch_start:batch_start + batch_size]
                try:
                    resp = requests.post(
                        f"{config.OLLAMA_BASE_URL}/api/embed",
                        json={
                            "model": config.OLLAMA_EMBED_MODEL,
                            "input": batch_texts,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for h, emb_list in zip(batch_h, data["embeddings"]):
                        new_embeds[h] = np.array(emb_list, dtype=np.float32)
                except Exception as e:
                    print(f"  Batch embedding error (batch {batch_start}): {e}")
                    dim = 768
                    for h in batch_h:
                        new_embeds[h] = np.zeros(dim, dtype=np.float32)

                done = min(batch_start + batch_size, n_total)
                if done % 200 == 0 or done == n_total:
                    print(f"  Embedded {done}/{n_total}")

        # --- merge cached + new, build result matrix -----------------------
        all_embeds = {**cached_embeds, **new_embeds}

        # Infer dimension from first vector
        dim = 768
        for v in all_embeds.values():
            dim = len(v)
            break

        result = np.zeros((len(texts_to_embed), dim), dtype=np.float32)
        for i, h in enumerate(hashes):
            if h in all_embeds:
                result[i] = all_embeds[h]

        # --- persist updated cache -----------------------------------------
        all_hashes = list(all_embeds.keys())
        all_vectors = np.array([all_embeds[h] for h in all_hashes], dtype=np.float32)
        full_hash_map = {h: idx for idx, h in enumerate(all_hashes)}
        np.save(self.cache_path, all_vectors)
        with open(self.hash_path, "w") as f:
            json.dump(full_hash_map, f)

        return result
    
    def build_dense_index(self):
        """Build or load the dense embedding index."""
        if self._embeddings is not None:
            return
        
        if self._load_cached_embeddings() and len(self._hash_map) >= len(self.texts):
            # Reconstruct embeddings in correct order
            cached_matrix = np.load(self.cache_path)
            self._embeddings = np.zeros((len(self.texts), cached_matrix.shape[1]), dtype=np.float32)
            for i, t in enumerate(self.texts):
                h = text_hash(t)
                if h in self._hash_map:
                    self._embeddings[i] = cached_matrix[int(self._hash_map[h])]
            return
        
        # Need to embed
        self._embeddings = self.embed_texts(self.texts)
    
    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity between a query vector and a matrix of doc vectors.

        Handles zero-norm vectors safely: any doc or query whose L2 norm is
        effectively zero gets similarity = 0.0 (no division warnings).
        """
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-12:
            return np.zeros(len(doc_vecs), dtype=np.float32)

        query_unit = query_vec / q_norm
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        # Mask zero-norm docs to avoid division warnings
        safe_norms = np.where(doc_norms < 1e-12, 1.0, doc_norms)
        doc_units = doc_vecs / safe_norms
        # Zero out rows that had zero norm
        doc_units[doc_norms.squeeze() < 1e-12] = 0.0
        return doc_units @ query_unit
    
    def search_bm25(self, query: str, top_k: int = 10) -> list:
        """BM25 search. Returns list of (id, score, rank)."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i]), rank + 1) 
                for rank, i in enumerate(top_indices) if scores[i] > 0]
    
    def search_dense(self, query: str, top_k: int = 10) -> list:
        """Dense vector search. Returns list of (id, score, rank)."""
        self.build_dense_index()
        query_vec = self.embed_texts([query])[0]
        scores = self.cosine_similarity(query_vec, self._embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i]), rank + 1)
                for rank, i in enumerate(top_indices)]
    
    def search_hybrid(self, query: str, top_k: int = 10, k: int = None) -> list:
        """Hybrid search with Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each retriever.
        
        Returns list of dicts with id, rrf_score, bm25_rank, bm25_score,
        dense_rank, dense_score.
        """
        k = k or config.RRF_K
        
        bm25_results = self.search_bm25(query, top_k=top_k * 3)
        
        # Try dense search, fall back to BM25-only if embeddings unavailable
        dense_results = []
        try:
            if self._embeddings is not None or self._load_cached_embeddings():
                dense_results = self.search_dense(query, top_k=top_k * 3)
        except Exception as e:
            dense_results = []
        
        # Build rank maps
        bm25_ranks = {r[0]: (r[2], r[1]) for r in bm25_results}
        dense_ranks = {r[0]: (r[2], r[1]) for r in dense_results}
        
        all_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        
        results = []
        for doc_id in all_ids:
            rrf_score = 0.0
            bm25_rank = bm25_ranks.get(doc_id, (None, 0.0))
            dense_rank = dense_ranks.get(doc_id, (None, 0.0))
            
            if bm25_rank[0] is not None:
                rrf_score += 1.0 / (k + bm25_rank[0])
            if dense_rank[0] is not None:
                rrf_score += 1.0 / (k + dense_rank[0])
            
            results.append({
                "id": doc_id,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank[0],
                "bm25_score": bm25_rank[1],
                "dense_rank": dense_rank[0],
                "dense_score": dense_rank[1],
            })
        
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]
    
    def evaluate_retrieval(self, labels: dict, top_k_values: list = [3, 5, 10]) -> dict:
        """Evaluate BM25-only vs vector-only vs RRF.
        
        Args:
            labels: dict mapping id -> category label.
            top_k_values: list of k values for recall@k.
        
        Returns:
            Dict with recall@k for each method.
        """
        results = {"bm25": {}, "dense": {}, "rrf": {}}
        
        sample_ids = [i for i in self.ids if i in labels][:100]
        
        for method in results:
            for top_k in top_k_values:
                hits = 0
                total = 0
                for doc_id in sample_ids:
                    idx = self.ids.index(doc_id)
                    query = self.texts[idx]
                    true_label = labels[doc_id]
                    
                    if method == "bm25":
                        res = self.search_bm25(query, top_k=top_k + 1)
                        retrieved_ids = [r[0] for r in res if r[0] != doc_id][:top_k]
                    elif method == "dense":
                        try:
                            res = self.search_dense(query, top_k=top_k + 1)
                            retrieved_ids = [r[0] for r in res if r[0] != doc_id][:top_k]
                        except:
                            continue
                    else:
                        res = self.search_hybrid(query, top_k=top_k + 1)
                        retrieved_ids = [r["id"] for r in res if r["id"] != doc_id][:top_k]
                    
                    # Check if any retrieved doc has same label
                    for rid in retrieved_ids:
                        if rid in labels and labels[rid] == true_label:
                            hits += 1
                            break
                    total += 1
                
                if total > 0:
                    results[method][f"recall@{top_k}"] = hits / total
        
        return results
