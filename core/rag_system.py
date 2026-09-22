"""
Main Configurable RAG System
"""

import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from typing import List, Optional, Tuple, Dict, Any

from .models import StrategyConfig, MixingConfig, MixingStrategy, STRATEGY_CONFIGS
from .question_generator import QuestionGenerator
from .document_retriever import DocumentRetriever
from .document_ranker import DocumentRanker
from config import (
    OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION, MODEL_NAME,
    ENABLE_QUERY_FILTERING
)


class ConfigurableRAGSystem:
    """Configurable RAG system with alternate questioning strategies"""
    
    def __init__(self, 
                 collection_name="configurable_rag", 
                 embedding_model_name="all-MiniLM-L6-v2",
                 strategy_config: Optional[StrategyConfig] = None,
                 mixing_config: Optional[MixingConfig] = None):
        """Initialize the configurable RAG system"""
        
        # Set default configurations
        self.strategy_config = strategy_config or STRATEGY_CONFIGS["S1"]
        self.mixing_config = mixing_config or MixingConfig(
            strategy=MixingStrategy.TOP_K_PER_SET,
            top_k_per_set=5,
            max_total_results=15
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB client (in-memory database)
        self.chroma_client = chromadb.Client()
        
        # Create a collection in ChromaDB
        try:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        except ValueError:
            # Collection already exists
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        
        # OpenAI API key and Azure endpoint from config
        self.api_key = OPENAI_API_KEY
        self.azure_endpoint = OPENAI_API_BASE
        self.azure_deployment = MODEL_NAME
        self.api_version = OPENAI_API_VERSION
        
        self.openai_client = None
        
        # Initialize Azure OpenAI client if API key is provided
        if self.api_key:
            try:
                if self.azure_endpoint.endswith('/'):
                    self.azure_endpoint = self.azure_endpoint[:-1]
                
                self.openai_client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint
                )
                print("Azure OpenAI client initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Azure OpenAI client: {e}")
        
        # Initialize component modules
        self.question_generator = QuestionGenerator(
            self.openai_client, 
            self.azure_deployment, 
            self.embedding_model
        )
        self.document_retriever = DocumentRetriever(
            self.collection, 
            self.embedding_model
        )
        self.document_ranker = DocumentRanker(self.embedding_model)
    
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> List[str]:
        """Embed documents and add to vector database"""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if len(doc_ids) != len(documents):
            raise ValueError("Number of document IDs must match number of documents")
        
        # Create embeddings
        print("Generating document embeddings...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=doc_ids
        )
        
        print(f"Added {len(documents)} documents to the vector database")
        return doc_ids
    
    def generate_alternate_questions(self, original_question: str) -> List[str]:
        """
        Generate alternate questions based on the configured strategy
        Returns a list of questions including the original
        """
        strategy = self.strategy_config.strategy
        query_set_size = self.strategy_config.query_set_size
        
        print(f"\nUsing strategy: {strategy.value} (Query Set Size: {query_set_size})")
        
        questions = self.question_generator.generate_single_strategy(
            original_question, strategy, query_set_size
        )
        
        print(f"\nGenerated {len(questions)} alternate question(s):")
        for i, q in enumerate(questions, 1):
            print(f"  Q{i}: {q}")
        
        return questions
    
    def generate_all_strategy_questions(self, original_question: str) -> Dict[str, List[str]]:
        """
        Generate alternate questions using ALL strategies (S1-S4)
        Returns a dictionary mapping strategy name to list of questions
        """
        return self.question_generator.generate_all_strategies(original_question)
    
    def query(self, user_question: str, top_k: int = 5, verbose: bool = True, 
              use_all_strategies: bool = False) -> Tuple[str, List[str]]:
        """
        Perform a configurable RAG query:
        1. Generate alternate questions based on strategy (or all strategies if use_all_strategies=True)
        2. Query vector DB for each alternate question
        3. Mix and sort results based on mixing config
        4. Generate final answer using LLM
        
        Args:
            user_question: The user's question
            top_k: Number of documents to retrieve per query
            verbose: Whether to print detailed output
            use_all_strategies: If True, use all strategies (S1-S4) instead of just configured strategy
        
        Returns:
            Tuple of (answer, list of context documents)
        """
        if verbose:
            print("\n" + "="*80)
            print(f"Processing query: {user_question}")
            print("="*80)
        
        # Analyze query type
        query_type = QuestionGenerator.analyze_query_type(user_question)
        if verbose:
            print(f"Query type detected: {query_type}")
        
        # Step 1: Generate alternate questions
        if use_all_strategies:
            # Use all strategies (S1-S4)
            all_strategy_questions = self.generate_all_strategy_questions(user_question)
            
            # Apply quality filtering to each strategy's questions
            if ENABLE_QUERY_FILTERING:
                for strategy_name, questions in all_strategy_questions.items():
                    if strategy_name != 'S1':  # Don't filter original question
                        filtered = self.question_generator.filter_quality_questions(
                            questions, user_question
                        )
                        all_strategy_questions[strategy_name] = filtered
            
            # Flatten all questions into a single list with strategy tags
            alternate_questions = []
            question_strategy_map = {}  # Track which strategy each question came from
            for strategy_name, questions in all_strategy_questions.items():
                for q in questions:
                    alternate_questions.append(q)
                    question_strategy_map[q] = strategy_name
        else:
            # Use only the configured strategy
            alternate_questions = self.generate_alternate_questions(user_question)
            question_strategy_map = {q: self.strategy_config.strategy.value for q in alternate_questions}
        
        # Step 2: Query vector DB for each alternate question
        if verbose:
            print(f"\nQuerying vector database for {len(alternate_questions)} question(s)...")
        
        result_sets = []
        for i, question in enumerate(alternate_questions, 1):
            if verbose:
                print(f"\n  Query {i}/{len(alternate_questions)}: {question}")
            
            # Get strategy for this question to use dynamic K
            strategy = question_strategy_map.get(question, None) if use_all_strategies else None
            result_set = self.document_retriever.query_vector_db(question, top_k=top_k, strategy=strategy)
            result_sets.append(result_set)
            
            if verbose:
                retrieved_count = len(result_set['documents'])
                print(f"    Retrieved {retrieved_count} documents (strategy: {strategy})")
        
        # Step 3: Mix and sort results
        mixed_results = self.document_retriever.mix_and_sort_results(result_sets, self.mixing_config)
        
        # Step 4: Apply advanced processing (reranking, MMR, optimization)
        mixed_results = self.document_ranker.apply_advanced_processing(user_question, mixed_results)
        
        if verbose:
            print("\n" + "-"*80)
            print("Mixed and sorted documents:")
            for i, result in enumerate(mixed_results, 1):
                print(f"{i}. {result['document'][:100]}...")
                print(f"   [Source: {result['source_question'][:60]}...]")
            print("-"*80)
        
        # Step 5: Generate final answer using LLM
        if not self.openai_client:
            context_docs = [r['document'] for r in mixed_results]
            context = "\n\n".join(context_docs)
            return f"""
No OpenAI API key provided. Cannot generate final answer.

Retrieved Context ({len(mixed_results)} documents):
{context}
""", context_docs
        
        # Build context from mixed results
        context_docs = [r['document'] for r in mixed_results]
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
        
        prompt = f"""Answer the following question based on the provided context documents.
If you cannot answer the question based on the context, say "I don't have enough information."

Context Documents:
{context}

Original Question: {user_question}

Answer:"""
        
        try:
            if verbose:
                print("\nGenerating final answer with LLM...")
            
            response = self.openai_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            if verbose:
                print("\n" + "="*80)
                print("FINAL ANSWER:")
                print("="*80)
            
            # Return answer and context documents
            context_docs = [r['document'] for r in mixed_results]
            return answer, context_docs
        
        except Exception as e:
            context_docs = [r['document'] for r in mixed_results]
            return f"Error generating final answer: {e}", context_docs
