"""
Document retrieval and result mixing logic
"""

from typing import List, Dict, Any, Optional
from .models import MixingStrategy
from config import TOP_K_RETRIEVAL, DEFAULT_TOP_K


class DocumentRetriever:
    """Handles vector database queries and result mixing"""
    
    def __init__(self, collection, embedding_model):
        """
        Initialize the document retriever
        
        Args:
            collection: ChromaDB collection instance
            embedding_model: SentenceTransformer model for embeddings
        """
        self.collection = collection
        self.embedding_model = embedding_model
    
    def query_vector_db(self, question: str, top_k: int = 5, strategy: str = None) -> Dict[str, Any]:
        """
        Query vector database for a single question
        
        Args:
            question: The question to search for
            top_k: Number of documents to retrieve
            strategy: Strategy name (S1, S2, S3, S4) for dynamic K
            
        Returns:
            Dictionary with documents, distances, ids, and metadata
        """
        # Use dynamic K based on strategy if configured
        if strategy and strategy in TOP_K_RETRIEVAL:
            top_k = TOP_K_RETRIEVAL.get(strategy, DEFAULT_TOP_K)
        
        # Embed the query
        query_embedding = self.embedding_model.encode(question).tolist()
        
        # Search for similar documents
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return {
            'question': question,
            'documents': results['documents'][0],
            'distances': results['distances'][0] if 'distances' in results else [],
            'ids': results['ids'][0],
            'strategy': strategy
        }
    
    def mix_and_sort_results(self, result_sets: List[Dict[str, Any]], 
                            mixing_config) -> List[Dict[str, Any]]:
        """
        Mix and sort results from multiple query result sets
        
        Args:
            result_sets: List of query results from different questions
            mixing_config: MixingConfig instance
            
        Returns:
            Deduplicated and sorted list of document dictionaries
        """
        strategy = mixing_config.strategy
        
        print(f"\nMixing results using strategy: {strategy.value}")
        
        mixed_results = []
        seen_docs = set()
        
        if strategy == MixingStrategy.TOP_K_PER_SET:
            # Take top K from each result set (best ranked by similarity)
            for result_set in result_sets:
                docs = result_set['documents']
                distances = result_set['distances']
                ids = result_set['ids']
                
                # Take top K documents from this result set
                for i in range(min(mixing_config.top_k_per_set, len(docs))):
                    doc = docs[i]
                    if mixing_config.remove_duplicates and doc in seen_docs:
                        continue
                    
                    seen_docs.add(doc)
                    mixed_results.append({
                        'document': doc,
                        'distance': distances[i] if distances else 0,
                        'id': ids[i],
                        'source_question': result_set['question']
                    })
        
        elif strategy == MixingStrategy.ROUND_ROBIN:
            # Alternate between result sets
            max_len = max(len(rs['documents']) for rs in result_sets)
            for i in range(max_len):
                for result_set in result_sets:
                    if i < len(result_set['documents']):
                        doc = result_set['documents'][i]
                        if mixing_config.remove_duplicates and doc in seen_docs:
                            continue
                        
                        seen_docs.add(doc)
                        distances = result_set['distances']
                        mixed_results.append({
                            'document': doc,
                            'distance': distances[i] if distances else 0,
                            'id': result_set['ids'][i],
                            'source_question': result_set['question']
                        })
        
        elif strategy == MixingStrategy.SCORE_BASED:
            # Collect all results with scores and sort by distance
            all_results = []
            for result_set in result_sets:
                docs = result_set['documents']
                distances = result_set['distances']
                ids = result_set['ids']
                
                for i in range(len(docs)):
                    all_results.append({
                        'document': docs[i],
                        'distance': distances[i] if distances else 0,
                        'id': ids[i],
                        'source_question': result_set['question']
                    })
            
            # Sort by distance (lower is better)
            all_results.sort(key=lambda x: x['distance'])
            
            # Deduplicate if needed
            for result in all_results:
                doc = result['document']
                if mixing_config.remove_duplicates and doc in seen_docs:
                    continue
                seen_docs.add(doc)
                mixed_results.append(result)
        
        elif strategy == MixingStrategy.DEDUPLICATE:
            # Simply deduplicate and merge all results
            for result_set in result_sets:
                docs = result_set['documents']
                distances = result_set['distances']
                ids = result_set['ids']
                
                for i in range(len(docs)):
                    doc = docs[i]
                    if doc in seen_docs:
                        continue
                    
                    seen_docs.add(doc)
                    mixed_results.append({
                        'document': doc,
                        'distance': distances[i] if distances else 0,
                        'id': ids[i],
                        'source_question': result_set['question']
                    })
        
        # Limit to max_total_results
        mixed_results = mixed_results[:mixing_config.max_total_results]
        
        print(f"Mixed results: {len(mixed_results)} documents (from {len(result_sets)} query sets)")
        
        return mixed_results
