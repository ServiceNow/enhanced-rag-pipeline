"""
Benchmark Runner
----------------
Run benchmarks and generate reports.
"""

import json
import math
import time
import shutil
import yaml
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict

import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .config import BenchmarkConfig, ChunkingStrategy
from config.strategy import MixingStrategy
from .beir_loader import BEIRLoader, Document
from evaluation.metrics import RetrievalMetrics, RetrievalMetricResult
from core.question_generator import QuestionGenerator
from core.models import QuestionStrategy, STRATEGY_CONFIGS


@dataclass
class QueryRetrievalResult:
    """Detailed retrieval results for a single query"""
    query_id: str
    original_query: str
    query_variations: List[str]  # Generated query variations
    retrieved_docs: List[Dict[str, Any]]  # [{doc_id, rank, similarity_score, is_relevant}]
    relevant_docs: List[str]  # Ground truth relevant doc IDs
    metrics: Dict[str, float]  # Per-query metrics


@dataclass
class BenchmarkResult:
    """Complete benchmark results"""
    config: Dict[str, Any]
    dataset_name: str
    num_queries: int
    num_documents: int
    metrics: Dict[str, Dict[int, RetrievalMetricResult]]  # {metric_name: {k: result}}
    runtime_seconds: float
    timestamp: str
    retrieval_details: Optional[List[QueryRetrievalResult]] = None  # Detailed per-query results


class BenchmarkRunner:
    """Run retrieval benchmarks with RAG strategy support"""
    
    def __init__(self, config: BenchmarkConfig, seed: Optional[int] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            seed: Random seed for reproducibility (default: None for non-deterministic)
        """
        self.config = config
        self.seed = seed
        self.embedding_model = None
        self.cross_encoder = None  # For reranking
        self.collection = None
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.question_generator = None
        self.openai_client = None
        self.retrieval_details = []  # Store detailed retrieval info
        self.config_path = None  # Will be set by run_benchmark.py

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Initialize LLM
        self.llm = None
        if self.config.strategy != "S1":
            self._init_llm()
    
    def _get_device(self) -> str:
        """Get the best available device for computations"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _init_llm(self) -> None:
        """Initialize LLM based on config"""
        llm_config = getattr(self.config, 'llm', None)
        
        if not llm_config:
            print("  ⚠️  No LLM configuration found")
            return
            
        llm_type = getattr(self.config, 'llm_type', 'gpt')
        
        try:
            if llm_type == "gpt":
                from core.llm.azure import AzureOpenAIModel
                self.llm = AzureOpenAIModel(
                    model_name_path=llm_config.model_name_path,
                    max_tokens=llm_config.max_tokens,
                    temperature=llm_config.temperature,
                    seed=self.seed,
                )
                seed_msg = f" (seed={self.seed})" if self.seed is not None else ""
                print(f"  ✓ Azure OpenAI LLM initialized: {llm_config.model_name_path}{seed_msg}")
            elif llm_type == "vllm":
                from core.llm.vllm import VLLMModel
                self.llm = VLLMModel(
                    model_name_path=llm_config.model_name_path,
                    max_tokens=llm_config.max_tokens,
                    temperature=llm_config.temperature,
                    gpu_memory_utilization=llm_config.gpu_memory_utilization,
                    max_model_len=llm_config.max_model_len
                )
                print(f"  ✓ VLLM initialized: {llm_config.model_name_path}")
            elif llm_type in ("openai", "openai_compatible"):
                from core.llm.openai_compatible import OpenAICompatibleModel
                if not llm_config.base_url:
                    raise ValueError("OpenAI-compatible backend requires llm.base_url")
                self.llm = OpenAICompatibleModel(
                    model_name_path=llm_config.model_name_path,
                    base_url=llm_config.base_url,
                    max_tokens=llm_config.max_tokens,
                    temperature=llm_config.temperature,
                    seed=self.seed,
                )
                print(f"  ✓ OpenAI-compatible LLM initialized: {llm_config.model_name_path}")
            else:
                raise ValueError(f"Unknown LLM type: {llm_type}")
        except Exception as e:
            if self.config.strict_llm:
                raise RuntimeError(f"Failed to initialize LLM: {e}") from e
            print(f"  ⚠️  Failed to initialize LLM: {e}")
            self.llm = None
        
    def _init_question_generator(self) -> None:
        """Initialize question generator using the integrated LLM"""
        if self.config.strategy == "S1":
            # S1 doesn't need LLM
            return
            
        if not self.llm:
            if self.config.strict_llm:
                raise RuntimeError(f"Strategy {self.config.strategy} requires an initialized LLM")
            print("  ⚠️  No LLM available - falling back to S1 (original query only)")
            self.config.strategy = "S1"
            return
        
        try:
            llm_config = self.config.llm
            self.question_generator = QuestionGenerator(
                self.llm,  # Use your LLM class instead of AzureOpenAI client
                llm_config.model_name_path,
                self.embedding_model,
                prompt_versions=self.config.prompt_config.versions,
                custom_prompt_paths=self.config.prompt_config.custom_paths
            )
            print(f"  ✓ Question generator initialized for strategy: {self.config.strategy}")
        except Exception as e:
            if self.config.strict_llm:
                raise RuntimeError(f"Failed to initialize question generator: {e}") from e
            print(f"  ⚠️  Failed to init question generator: {e} - falling back to S1")
            self.config.strategy = "S1"
        
    def run(self) -> BenchmarkResult:
        """
        Run the complete benchmark.
        
        Returns:
            BenchmarkResult with all metrics
        """
        start_time = time.time()
        
        # Load data
        print(f"\n{'='*60}")
        print("BENCHMARK: Loading data...")
        print(f"{'='*60}")
        loader = BEIRLoader(self.config.dataset)
        self.corpus, self.queries, self.qrels = loader.load()
        
        # Initialize embedding model
        print(f"\nLoading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model,
            device=self._get_device(),
            cache_folder='./embeddings_cache',  # Cache for faster loading
            trust_remote_code=False
        )
        
        # Optimize embedding model for performance
        if hasattr(self.embedding_model, 'max_seq_length'):
            print(f"  Max sequence length: {self.embedding_model.max_seq_length}")
        
        # Pre-warm the model with a dummy encode (faster first real use)
        print("  Pre-warming embedding model...")
        self.embedding_model.encode(["warm up"], show_progress_bar=False)
        print("  ✓ Embedding model ready")
        
        # Initialize cross-encoder for reranking
        if self.config.use_reranking:
            print(f"Loading rerank model: {self.config.rerank_model}")
            self.cross_encoder = CrossEncoder(self.config.rerank_model)
        
        # Initialize question generator for strategies
        print(f"\nInitializing strategy: {self.config.strategy}")
        self._init_question_generator()
        
        # Prepare documents (with optional chunking)
        documents, doc_ids = self._prepare_documents()
        
        # Index documents
        print(f"\nIndexing {len(documents)} documents...")
        self._index_documents(documents, doc_ids)
        
        # Run retrieval for all queries
        evaluation_query_count = sum(query_id in self.qrels for query_id in self.queries)
        if self.config.max_queries is not None:
            evaluation_query_count = min(evaluation_query_count, self.config.max_queries)
        print(f"\nRunning retrieval (strategy={self.config.strategy}) for {evaluation_query_count} queries...")
        all_retrieved = self._retrieve_all()
        
        # Compute metrics
        print(f"\nComputing metrics...")
        metrics = self._compute_metrics(all_retrieved)
        
        runtime = time.time() - start_time
        
        # Build result
        result = BenchmarkResult(
            config=self._config_to_dict(),
            dataset_name=self.config.dataset.name,
            num_queries=len(all_retrieved),
            num_documents=len(self.corpus),
            metrics=metrics,
            runtime_seconds=runtime,
            timestamp=datetime.now().isoformat(),
            retrieval_details=self.retrieval_details if self.retrieval_details else None
        )
        
        # Print summary
        self._print_summary(result)
        
        # Save results with config file
        self._save_results(result, self.config_path)
        
        return result
    
    def _prepare_documents(self) -> tuple[List[str], List[str]]:
        """Prepare documents, optionally chunking them"""
        documents = []
        doc_ids = []
        
        # Sort corpus by doc_id for deterministic order
        sorted_corpus = sorted(self.corpus.items(), key=lambda x: x[0])
        
        if self.config.chunking.strategy == ChunkingStrategy.NONE:
            # No chunking - use documents as-is
            for doc_id, doc in sorted_corpus:
                documents.append(doc.text)
                doc_ids.append(doc_id)
        else:
            # Apply chunking
            print(f"Chunking documents with strategy: {self.config.chunking.strategy.value}")
            for doc_id, doc in sorted_corpus:
                chunks = self._chunk_document(doc.text)
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    doc_ids.append(f"{doc_id}_chunk_{i}")
        
        print(f"  Prepared {len(documents)} document segments")
        return documents, doc_ids
    
    def _chunk_document(self, text: str) -> List[str]:
        """Chunk a document based on config"""
        strategy = self.config.chunking.strategy
        chunk_size = self.config.chunking.chunk_size
        overlap = self.config.chunking.chunk_overlap
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, chunk_size)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, chunk_size)
        else:
            return [text]
    
    def _chunk_fixed_size(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fixed-size chunking with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else len(text)
        return chunks if chunks else [text]
    
    def _chunk_by_sentence(self, text: str, max_chunk_size: int) -> List[str]:
        """Sentence-based chunking with a hard char cap.

        Long inputs without punctuation (chat snippets, code blocks, JSON
        dumps, log lines) used to produce single chunks tens of KB long,
        which then blew up the tokenizer on large corpora. We now split
        any sentence that already exceeds max_chunk_size at the nearest
        whitespace boundary so every chunk fits.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        def split_long(s: str, cap: int) -> List[str]:
            if len(s) <= cap:
                return [s]
            pieces = []
            i = 0
            while i < len(s):
                end = min(i + cap, len(s))
                if end < len(s):
                    space = s.rfind(' ', i + cap // 2, end)
                    if space > i:
                        end = space
                pieces.append(s[i:end].strip())
                i = end
            return [p for p in pieces if p]

        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            for piece in split_long(sent, max_chunk_size):
                if current_len + len(piece) > max_chunk_size and current:
                    chunks.append(' '.join(current))
                    current = []
                    current_len = 0
                current.append(piece)
                current_len += len(piece) + 1

        if current:
            chunks.append(' '.join(current))

        return chunks if chunks else [text]
    
    def _chunk_by_paragraph(self, text: str, max_chunk_size: int) -> List[str]:
        """Paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > max_chunk_size and current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para) + 2
        
        if current:
            chunks.append('\n\n'.join(current))
        
        return chunks if chunks else [text]
    
    def _index_documents(self, documents: List[str], doc_ids: List[str]) -> None:
        """Index documents into ChromaDB"""
        client = chromadb.Client()
        
        # Create deterministic collection name if seed is set
        if self.seed is not None:
            collection_name = f"benchmark_seed_{self.seed}"
        else:
            collection_name = f"benchmark_{int(time.time())}"
        
        # Delete collection if it exists (for deterministic reruns)
        try:
            client.delete_collection(name=collection_name)
        except:
            pass
        
        # Use HNSW for fast approximate search
        print("  Using HNSW index for fast approximate search")
        metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:M": 16,
        }
        
        self.collection = client.create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        # Embed and add in batches - maximum GPU optimization
        device = self._get_device()
        
        # Aggressive batch sizes for maximum GPU utilization
        if device == 'cuda':
            batch_size = 500        # Much larger batches for GPU
            encode_batch_size = 256  # Max GPU batch size
        else:
            batch_size = 200        # Much larger CPU batches to reduce DB overhead
            encode_batch_size = 64   # Larger CPU encode batch for better efficiency
        
        print(f"  Using {device}-optimized batch sizes: {batch_size}/{encode_batch_size}")
        num_batches = (len(documents) + batch_size - 1) // batch_size

        # Stream each batch into ChromaDB immediately. Accumulating all
        # embeddings + chunk text in Python lists can exceed 40GB on
        # large corpora before any DB write happens.
        max_chroma_batch = 5000  # Stay under ChromaDB limit of 5461

        def flush_to_chroma(ids, docs, embs):
            for j in range(0, len(embs), max_chroma_batch):
                end_idx = min(j + max_chroma_batch, len(embs))
                self.collection.add(
                    embeddings=embs[j:end_idx],
                    documents=docs[j:end_idx],
                    ids=ids[j:end_idx],
                )

        for i in tqdm(range(0, len(documents), batch_size),
                      total=num_batches, desc="Encoding", unit="batch"):
            batch_docs = documents[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]

            embeddings = self.embedding_model.encode(
                batch_docs,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=device
            ).tolist()

            flush_to_chroma(batch_ids, batch_docs, embeddings)

        print(f"  Indexed {len(documents)} chunks into ChromaDB")


    def _retrieve_all(self) -> Dict[str, List[str]]:
        """Retrieve documents for all queries using configured strategy and mixing"""
        max_k = max(self.config.top_k_values)
        all_retrieved = {}
        all_scores = {}  # Track scores for retrieval details
        mixing = self.config.mixing_strategy
        
        # Filter to queries with qrels and sort for deterministic order
        valid_queries = [(qid, q) for qid, q in self.queries.items() if qid in self.qrels]
        valid_queries.sort(key=lambda x: x[0])  # Sort by query_id
        if self.config.max_queries is not None:
            valid_queries = valid_queries[:self.config.max_queries]
        
        # Clear previous retrieval details
        self.retrieval_details = []
        
        for query_id, query in tqdm(valid_queries, desc="Retrieving", unit="query"):
            # Generate query variations based on strategy
            query_variations = self._generate_query_variations(query.text)
            
            # Retrieve for each variation
            variation_results = []  # List of (doc_id, distance) tuples per variation
            doc_scores = {}  # Track best score for each doc
            
            # Maximum speed query encoding
            device = self._get_device()
            
            # Aggressive query batching for GPU
            if device == 'cuda':
                query_batch_size = 64  # Larger query batches
            else:
                query_batch_size = 16  # CPU-friendly
            
            if query_variations:
                all_query_texts = [q_text for q_text in query_variations]
                all_query_embeddings = self.embedding_model.encode(
                    all_query_texts,
                    batch_size=query_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device
                ).tolist()
                
                # Process each query variation
                for idx, q_text in enumerate(query_variations):
                    query_embedding = all_query_embeddings[idx]
                    
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max_k,
                        include=['distances']
                    )
                    
                    var_docs = []
                    for doc_id, dist in zip(results['ids'][0], results['distances'][0]):
                        parent_id = doc_id.split('_chunk_')[0]
                        var_docs.append((parent_id, dist))
                        # Track best (lowest) distance for each doc
                        # Store as tuple: (score, is_reranking_score)
                        if parent_id not in doc_scores:
                            doc_scores[parent_id] = (dist, False)
                        else:
                            # Only update if this is a better (lower) distance and not already reranked
                            current_score, is_reranked = doc_scores[parent_id]
                            if not is_reranked and dist < current_score:
                                doc_scores[parent_id] = (dist, False)
                    variation_results.append(var_docs)
            
            # Mix results based on strategy
            mixed_docs = self._mix_results(variation_results, max_k * 2, mixing)  # Get more for reranking
            
            # Apply reranking if enabled
            if self.config.use_reranking and self.cross_encoder and mixed_docs:
                mixed_docs, rerank_scores = self._rerank_documents_with_scores(query.text, mixed_docs, max_k * 2)
                # Update scores with reranking scores (mark as reranking scores with negative flag)
                for doc_id, score in zip(mixed_docs, rerank_scores):
                    # Store as tuple: (score, is_reranking_score)
                    doc_scores[doc_id] = (score, True)
            
            # Apply MMR if enabled
            if self.config.use_mmr and mixed_docs:
                mixed_docs = self._apply_mmr(query.text, mixed_docs, max_k)
            
            # Ensure we don't exceed max_k
            final_docs = mixed_docs[:max_k] if mixed_docs else []
            all_retrieved[query_id] = final_docs
            all_scores[query_id] = doc_scores
            
            # Capture detailed retrieval information with scores
            self._capture_retrieval_details(query_id, query.text, query_variations, final_docs, doc_scores)
        
        return all_retrieved
    
    def _capture_retrieval_details(self, query_id: str, original_query: str, 
                                   query_variations: List[str], retrieved_docs: List[str],
                                   doc_scores: Dict[str, float]) -> None:
        """Capture detailed retrieval information for analysis"""
        # Get relevant docs from qrels
        relevant_docs = []
        if query_id in self.qrels:
            relevant_docs = [doc_id for doc_id, score in self.qrels[query_id].items() if score > 0]
        
        # Build detailed doc info with similarity scores
        detailed_docs = []
        for rank, doc_id in enumerate(retrieved_docs, 1):
            # Get score and determine if it's from reranking
            score_data = doc_scores.get(doc_id, (float('inf'), False))
            if isinstance(score_data, tuple):
                score, is_reranked = score_data
            else:
                # Fallback for old format
                score, is_reranked = score_data, False
            
            # Convert to similarity score
            if is_reranked:
                # Reranking scores are already similarity-like (higher is better)
                # Normalize to 0-1 range using sigmoid
                similarity = 1.0 / (1.0 + math.exp(-score))
            else:
                # Distance scores: convert using 1 / (1 + distance)
                similarity = 1.0 / (1.0 + score) if score != float('inf') else 0.0
            
            doc_info = {
                'doc_id': doc_id,
                'rank': rank,
                'similarity_score': float(similarity),
                'is_relevant': doc_id in relevant_docs
            }
            detailed_docs.append(doc_info)
        
        # Compute metrics for this query at max_k
        max_k = max(self.config.top_k_values)
        qrels_dict = self.qrels.get(query_id, {})
        relevant_set = set(relevant_docs)
        
        query_metrics = RetrievalMetrics.compute_all(
            retrieved_docs, relevant_set, qrels_dict, max_k
        )
        
        # Store retrieval details
        detail = QueryRetrievalResult(
            query_id=query_id,
            original_query=original_query,
            query_variations=query_variations,
            retrieved_docs=detailed_docs,
            relevant_docs=relevant_docs,
            metrics=query_metrics
        )
        self.retrieval_details.append(detail)
    
    def _rerank_documents(self, query: str, doc_ids: List[str], top_k: int) -> List[str]:
        """Rerank documents using cross-encoder"""
        reranked, _ = self._rerank_documents_with_scores(query, doc_ids, top_k)
        return reranked
    
    def _rerank_documents_with_scores(self, query: str, doc_ids: List[str], top_k: int) -> tuple[List[str], List[float]]:
        """Rerank documents using cross-encoder and return scores"""
        if len(doc_ids) <= 1:
            return doc_ids, [1.0] * len(doc_ids)
        
        # Get document texts
        doc_texts = []
        valid_ids = []
        for doc_id in doc_ids:
            if doc_id in self.corpus:
                doc_texts.append(self.corpus[doc_id].text)
                valid_ids.append(doc_id)
        
        if not doc_texts:
            return doc_ids, [0.0] * len(doc_ids)
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in doc_texts]
        
        # Get relevance scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by scores (descending)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc_id for doc_id, _ in ranked[:top_k]]
        top_scores = [score for _, score in ranked[:top_k]]
        return top_docs, top_scores
    
    def _apply_mmr(self, query: str, doc_ids: List[str], top_k: int) -> List[str]:
        """Apply Maximal Marginal Relevance for diversity"""
        if len(doc_ids) <= top_k:
            return doc_ids
        
        # Get document texts and embeddings
        doc_texts = []
        valid_ids = []
        for doc_id in doc_ids:
            if doc_id in self.corpus:
                doc_texts.append(self.corpus[doc_id].text)
                valid_ids.append(doc_id)
        
        if len(valid_ids) <= top_k:
            return valid_ids
        
        # Encode query and documents
        query_emb = self.embedding_model.encode([query])
        doc_embs = self.embedding_model.encode(doc_texts)
        
        # Calculate relevance scores
        relevance_scores = cosine_similarity(doc_embs, query_emb).flatten()
        
        # MMR selection
        lambda_param = self.config.mmr_lambda
        selected_indices = []
        remaining_indices = list(range(len(valid_ids)))
        
        # Select first document (most relevant)
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents balancing relevance and diversity
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = relevance_scores[idx]
                
                # Max similarity to already selected documents
                selected_embs = doc_embs[selected_indices]
                similarities = cosine_similarity(
                    doc_embs[idx].reshape(1, -1),
                    selected_embs
                ).flatten()
                max_sim = np.max(similarities) if len(similarities) > 0 else 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [valid_ids[i] for i in selected_indices]
    
    def _mix_results(self, variation_results: List[List[tuple]], max_k: int, 
                     mixing: MixingStrategy) -> List[str]:
        """Mix results from multiple query variations"""
        
        if mixing == MixingStrategy.SCORE_BASED:
            # Rank by best (lowest) distance score
            doc_scores = {}  # doc_id -> best score
            for var_docs in variation_results:
                for doc_id, dist in var_docs:
                    if doc_id not in doc_scores or dist < doc_scores[doc_id]:
                        doc_scores[doc_id] = dist
            
            # Sort by score (lower distance = better)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1])
            return [doc_id for doc_id, _ in sorted_docs[:max_k]]
        
        elif mixing == MixingStrategy.ROUND_ROBIN:
            # Alternate between variation results
            result = []
            seen = set()
            max_len = max(len(vr) for vr in variation_results) if variation_results else 0
            
            for i in range(max_len):
                for var_docs in variation_results:
                    if i < len(var_docs):
                        doc_id = var_docs[i][0]
                        if doc_id not in seen:
                            result.append(doc_id)
                            seen.add(doc_id)
                            if len(result) >= max_k:
                                return result
            return result[:max_k]
        
        else:  # DEDUPLICATE or TOP_K_PER_SET (simple dedup)
            result = []
            seen = set()
            for var_docs in variation_results:
                for doc_id, _ in var_docs:
                    if doc_id not in seen:
                        result.append(doc_id)
                        seen.add(doc_id)
                        if len(result) >= max_k:
                            return result
            return result[:max_k]
    
    def _generate_query_variations(self, query_text: str) -> List[str]:
        """Generate query variations based on configured strategy"""
        strategy = self.config.strategy

        # S1 or no generator available - just use original
        if strategy == "S1" or not self.question_generator:
            return [query_text]

        # HyDE / Query2Doc are single-query baselines that REPLACE the original
        # query with a pseudo-document, rather than augmenting it with variations.
        # This matches the canonical definitions in Gao et al. (2022) and
        # Wang et al. (2023): retrieval is performed using only the generated
        # passage (HyDE) or the query-concatenated-with-passage (Query2Doc).
        if strategy in ("HyDE", "Query2Doc"):
            strat_config = STRATEGY_CONFIGS.get(strategy)
            if not strat_config:
                return [query_text]
            try:
                replacement = self.question_generator.generate_single_strategy(
                    query_text,
                    strat_config.strategy,
                    strat_config.query_set_size,
                )
                return replacement if replacement else [query_text]
            except Exception:
                if self.config.strict_llm:
                    raise
                return [query_text]

        variations = [query_text]  # Always include original

        try:
            if strategy == "all":
                # Generate for all strategies
                all_questions = self.question_generator.generate_all_strategies(query_text)
                for strat_name, questions in all_questions.items():
                    variations.extend(questions)
            else:
                # Single multi-query strategy (S2, S3, S4)
                strat_config = STRATEGY_CONFIGS.get(strategy)
                if strat_config:
                    questions = self.question_generator.generate_single_strategy(
                        query_text,
                        strat_config.strategy,
                        strat_config.query_set_size
                    )
                    variations.extend(questions)
        except Exception:
            if self.config.strict_llm:
                raise

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for v in variations:
            if v not in seen:
                unique.append(v)
                seen.add(v)

        return unique
    
    def _normalize_doc_id(self, doc_id: str) -> str:
        """Normalize chunk ID back to original document ID.
        
        Handles: 'doc_123_chunk_0' -> 'doc_123'
        """
        if '_chunk_' in doc_id:
            return doc_id.split('_chunk_')[0]
        return doc_id
    
    def _compute_metrics(self, all_retrieved: Dict[str, List[str]]) -> Dict[str, Dict[int, RetrievalMetricResult]]:
        """Compute all metrics for all k values"""
        metrics = {metric: {} for metric in self.config.metrics}
        
        # Check if chunking is enabled
        using_chunks = self.config.chunking.strategy != ChunkingStrategy.NONE
        
        for k in self.config.top_k_values:
            # Compute per-query metrics
            per_query_results = {}
            
            for query_id, retrieved in all_retrieved.items():
                if query_id not in self.qrels:
                    continue
                
                qrels = self.qrels[query_id]
                relevant = set(doc_id for doc_id, score in qrels.items() if score > 0)
                
                # If using chunks, normalize retrieved IDs to match qrels
                if using_chunks:
                    retrieved_normalized = [self._normalize_doc_id(doc_id) for doc_id in retrieved]
                    # Also create normalized qrels for NDCG calculation
                    qrels_normalized = {}
                    for doc_id, score in qrels.items():
                        norm_id = self._normalize_doc_id(doc_id)
                        qrels_normalized[norm_id] = max(qrels_normalized.get(norm_id, 0), score)
                else:
                    retrieved_normalized = retrieved
                    qrels_normalized = qrels
                
                per_query_results[query_id] = RetrievalMetrics.compute_all(
                    retrieved_normalized, relevant, qrels_normalized, k
                )
            
            # Aggregate
            for metric in self.config.metrics:
                metrics[metric][k] = RetrievalMetrics.aggregate(
                    per_query_results, metric, k
                )
        
        return metrics
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dict"""
        ds = self.config.dataset
        return {
            'dataset_name': ds.name,
            'corpus_source': ds.corpus.dataset or ds.corpus.path,
            'queries_source': ds.queries.dataset or ds.queries.path,
            'qrels_source': ds.qrels.dataset or ds.qrels.path,
            'embedding_model': self.config.embedding_model,
            'llm_type': self.config.llm_type,
            'llm_model': self.config.llm.model_name_path,
            'strategy': self.config.strategy,
            'mixing_strategy': self.config.mixing_strategy.value,
            'use_reranking': self.config.use_reranking,
            'use_mmr': self.config.use_mmr,
            'chunking_strategy': self.config.chunking.strategy.value,
            'chunk_size': self.config.chunking.chunk_size,
            'chunk_overlap': self.config.chunking.chunk_overlap,
            'top_k_values': self.config.top_k_values,
            'max_queries': self.config.max_queries,
            'metrics': self.config.metrics
        }
    
    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Dataset: {result.dataset_name}")
        print(f"Documents: {result.num_documents}")
        print(f"Queries: {result.num_queries}")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"\nMetrics:")
        
        # Print table header
        k_values = self.config.top_k_values
        header = f"{'Metric':<12}" + "".join(f"@{k:<7}" for k in k_values)
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for metric_name in self.config.metrics:
            row = f"{metric_name.upper():<12}"
            for k in k_values:
                if k in result.metrics[metric_name]:
                    value = result.metrics[metric_name][k].mean
                    row += f"{value:<8.4f}"
                else:
                    row += f"{'N/A':<8}"
            print(row)
        
        print(f"{'='*60}\n")
    
    def _save_results(self, result: BenchmarkResult, config_path: Optional[str] = None) -> None:
        """Save comprehensive results to structured output directory"""
        # Create timestamped output directory with nested structure: dataset_name/strategy_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.strategy}_{timestamp}"
        if self.config.run_suffix:
            run_name = f"{run_name}_{self.config.run_suffix}"
        if self.seed is not None:
            run_name = f"{run_name}_seed{self.seed}"
        output_dir = Path(self.config.output_dir) / result.dataset_name / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to: {output_dir}")
        
        # 1. Copy configuration file
        if config_path and Path(config_path).exists():
            config_copy = output_dir / "config.yaml"
            shutil.copy2(config_path, config_copy)
            print(f"  ✓ Configuration copied: config.yaml")
        
        # 2. Save configuration as JSON (for programmatic access)
        config_json = output_dir / "config.json"
        with open(config_json, 'w') as f:
            json.dump(result.config, f, indent=2)
        print(f"  ✓ Configuration saved: config.json")
        
        # 3. Save aggregated metrics
        metrics_data = {
            'dataset_name': result.dataset_name,
            'num_queries': result.num_queries,
            'num_documents': result.num_documents,
            'runtime_seconds': result.runtime_seconds,
            'timestamp': result.timestamp,
            'seed': self.seed,
            'strategy': self.config.strategy,
            'embedding_model': self.config.embedding_model,
            'metrics': {}
        }
        
        for metric_name, k_results in result.metrics.items():
            metrics_data['metrics'][metric_name] = {}
            for k, metric_result in k_results.items():
                metrics_data['metrics'][metric_name][str(k)] = {
                    'mean': metric_result.mean,
                    'std': metric_result.std
                }
        
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"  ✓ Metrics saved: metrics.json")
        
        # 4. Save detailed retrieval results
        if result.retrieval_details:
            retrieval_file = output_dir / "retrieval_details.jsonl"
            with open(retrieval_file, 'w') as f:
                for detail in result.retrieval_details:
                    f.write(json.dumps(asdict(detail)) + '\n')
            print(f"  ✓ Retrieval details saved: retrieval_details.jsonl ({len(result.retrieval_details)} queries)")
        
        # 5. Save per-query metrics (if enabled)
        if self.config.save_per_query:
            per_query_file = output_dir / "per_query_metrics.json"
            per_query_data = {}
            
            for metric_name, k_results in result.metrics.items():
                per_query_data[metric_name] = {}
                for k, metric_result in k_results.items():
                    if metric_result.per_query:
                        per_query_data[metric_name][str(k)] = metric_result.per_query
            
            with open(per_query_file, 'w') as f:
                json.dump(per_query_data, f, indent=2)
            print(f"  ✓ Per-query metrics saved: per_query_metrics.json")
        
        # 6. Create summary README
        readme_file = output_dir / "README.md"
        self._create_readme(readme_file, result)
        print(f"  ✓ Summary created: README.md")
        
        print(f"\n✓ All results saved to: {output_dir}")
    
    def _create_readme(self, path: Path, result: BenchmarkResult) -> None:
        """Create a human-readable summary README"""
        with open(path, 'w') as f:
            f.write(f"# Benchmark Results: {result.dataset_name}\n\n")
            f.write(f"**Timestamp:** {result.timestamp}\n\n")
            f.write(f"**Strategy:** {self.config.strategy}\n\n")
            f.write(f"**Runtime:** {result.runtime_seconds:.2f}s\n\n")
            
            f.write(f"## Dataset\n\n")
            f.write(f"- Documents: {result.num_documents}\n")
            f.write(f"- Queries: {result.num_queries}\n")
            f.write(f"- Embedding Model: {self.config.embedding_model}\n\n")
            
            f.write(f"## Configuration\n\n")
            f.write(f"- Strategy: {self.config.strategy}\n")
            f.write(f"- Mixing: {self.config.mixing_strategy.value}\n")
            f.write(f"- Reranking: {self.config.use_reranking}\n")
            f.write(f"- MMR: {self.config.use_mmr}\n")
            f.write(f"- Chunking: {self.config.chunking.strategy.value}\n")
            if self.config.chunking.strategy != ChunkingStrategy.NONE:
                f.write(f"  - Chunk Size: {self.config.chunking.chunk_size}\n")
                f.write(f"  - Overlap: {self.config.chunking.chunk_overlap}\n")
            f.write(f"\n")
            
            f.write(f"## Metrics\n\n")
            f.write(f"| Metric | " + " | ".join(f"@{k}" for k in self.config.top_k_values) + " |\n")
            f.write(f"|--------|" + "|".join("-------" for _ in self.config.top_k_values) + "|\n")
            
            for metric_name in self.config.metrics:
                row = f"| {metric_name.upper()} | "
                values = []
                for k in self.config.top_k_values:
                    if k in result.metrics[metric_name]:
                        value = result.metrics[metric_name][k].mean
                        values.append(f"{value:.4f}")
                    else:
                        values.append("N/A")
                row += " | ".join(values) + " |\n"
                f.write(row)
            
            f.write(f"\n## Files\n\n")
            f.write(f"- `config.yaml` - Original configuration file\n")
            f.write(f"- `config.json` - Configuration in JSON format\n")
            f.write(f"- `metrics.json` - Aggregated metrics results\n")
            f.write(f"- `retrieval_details.jsonl` - Per-query retrieval details\n")
            if self.config.save_per_query:
                f.write(f"- `per_query_metrics.json` - Per-query metric scores\n")
            f.write(f"- `README.md` - This file\n")
