"""
Document ranking and optimization logic
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from config import (
    ENABLE_RERANKING, RERANK_TOP_N, RERANK_MODEL,
    ENABLE_MMR, MMR_LAMBDA
)


class DocumentRanker:
    """Handles document reranking, MMR, and context optimization"""
    
    def __init__(self, embedding_model):
        """
        Initialize the document ranker
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
        """
        self.embedding_model = embedding_model
    
    def rerank_documents(self, query: str, documents: List[str], top_k: int = 20) -> List[int]:
        """
        Rerank documents using cross-encoder for better relevance scoring
        
        Args:
            query: The query text
            documents: List of document texts
            top_k: Number of top documents to return
        
        Returns:
            List of reranked document indices
        """
        if not ENABLE_RERANKING:
            return list(range(min(top_k, len(documents))))
        
        if len(documents) <= 1:
            return list(range(len(documents)))
        
        try:
            print(f"\n🔄 Reranking {len(documents)} documents with cross-encoder...")
            
            # Initialize cross-encoder
            cross_encoder = CrossEncoder(RERANK_MODEL)
            
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get relevance scores
            scores = cross_encoder.predict(pairs)
            
            # Sort by scores (descending)
            ranked_indices = np.argsort(scores)[::-1][:top_k].tolist()
            
            print(f"  ✓ Reranked to top {min(top_k, len(ranked_indices))} documents")
            
            return ranked_indices
        
        except Exception as e:
            print(f"  ⚠️ Reranking failed: {e}. Using original order.")
            return list(range(min(top_k, len(documents))))
    
    def apply_mmr(self, documents: List[str], embeddings: np.ndarray, 
                  query_embedding: np.ndarray, lambda_param: float = 0.7, 
                  top_k: int = 20) -> List[int]:
        """
        Apply Maximal Marginal Relevance for diversity-aware selection
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings (n_docs x embedding_dim)
            query_embedding: Query embedding (1 x embedding_dim)
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
            top_k: Number of documents to select
        
        Returns:
            List of selected document indices
        """
        if len(documents) <= top_k:
            return list(range(len(documents)))
        
        # Calculate relevance scores (similarity to query)
        relevance_scores = cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()
        
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (most relevant)
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents balancing relevance and diversity
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Max similarity to already selected documents (for diversity)
                selected_embeddings = embeddings[selected_indices]
                similarities = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    selected_embeddings
                ).flatten()
                max_sim = np.max(similarities) if len(similarities) > 0 else 0
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def optimize_context_window(self, documents: List[str], max_tokens: int = 3000) -> List[str]:
        """
        Optimize documents to fit within context window
        
        Args:
            documents: List of document texts
            max_tokens: Maximum number of tokens allowed
        
        Returns:
            Optimized list of documents
        """
        # Simple token estimation: ~4 characters per token
        total_chars = sum(len(doc) for doc in documents)
        estimated_tokens = total_chars / 4
        
        if estimated_tokens <= max_tokens:
            return documents
        
        print(f"\n✂️ Optimizing context window (estimated {estimated_tokens:.0f} tokens, max {max_tokens})...")
        
        # Truncate documents proportionally
        target_chars = max_tokens * 4
        scale_factor = target_chars / total_chars
        
        optimized_docs = []
        for doc in documents:
            max_doc_len = int(len(doc) * scale_factor)
            if len(doc) > max_doc_len:
                optimized_docs.append(doc[:max_doc_len] + "...")
            else:
                optimized_docs.append(doc)
        
        print(f"  ✓ Optimized to ~{max_tokens} tokens")
        return optimized_docs
    
    def apply_advanced_processing(self, query: str, mixed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply advanced processing: reranking, MMR, context optimization
        
        Args:
            query: The user query
            mixed_results: Mixed document results
            
        Returns:
            Processed document results
        """
        if not mixed_results:
            return mixed_results
        
        documents = [r['document'] for r in mixed_results]
        
        # Step 1: Reranking with cross-encoder
        if ENABLE_RERANKING:
            reranked_indices = self.rerank_documents(query, documents, top_k=RERANK_TOP_N)
            mixed_results = [mixed_results[i] for i in reranked_indices]
            documents = [documents[i] for i in reranked_indices]
        
        # Step 2: Apply MMR for diversity
        if ENABLE_MMR and len(documents) > 1:
            print(f"\n🎯 Applying MMR for diversity (λ={MMR_LAMBDA})...")
            
            # Get embeddings for documents and query
            doc_embeddings = self.embedding_model.encode(documents)
            query_embedding = self.embedding_model.encode([query])[0]
            
            mmr_indices = self.apply_mmr(
                documents,
                doc_embeddings,
                query_embedding,
                lambda_param=MMR_LAMBDA,
                top_k=min(20, len(documents))
            )
            
            mixed_results = [mixed_results[i] for i in mmr_indices]
            documents = [documents[i] for i in mmr_indices]
            print(f"  ✓ Selected {len(mmr_indices)} diverse documents")
        
        # Step 3: Optimize context window
        optimized_docs = self.optimize_context_window(documents, max_tokens=3000)
        for i, doc in enumerate(optimized_docs):
            mixed_results[i]['document'] = doc
        
        return mixed_results
