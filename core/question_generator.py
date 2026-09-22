"""
Question generation and reformulation logic
"""

from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .models import QuestionStrategy
from config import (
    ENABLE_QUERY_FILTERING,
    MIN_SIMILARITY_TO_ORIGINAL,
    MAX_SIMILARITY_TO_ORIGINAL,
    PROMPT_PATHS,
    DEFAULT_PROMPT_VERSIONS,
    _build_prompt_paths
)


class QuestionGenerator:
    """Handles question reformulation and quality filtering"""
    
    def __init__(self, llm, model_name, embedding_model, 
                 prompt_versions: Optional[Dict[str, str]] = None,
                 custom_prompt_paths: Optional[Dict[str, str]] = None):
        """
        Initialize the question generator
        
        Args:
            llm: LLM instance (AzureOpenAIModel or VLLMModel)
            model_name: Model name/deployment name
            embedding_model: SentenceTransformer model for embeddings
            prompt_versions: Dict of versions per strategy (e.g., {"neighbor": "v1", "synonyms": "v2"})
            custom_prompt_paths: Optional dict to override specific prompt paths
        """
        self.llm = llm
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.prompt_versions = prompt_versions or DEFAULT_PROMPT_VERSIONS
        self.custom_prompt_paths = custom_prompt_paths
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from external files."""
        # Build paths based on versions
        if self.custom_prompt_paths:
            prompt_paths = self.custom_prompt_paths
        else:
            prompt_paths = _build_prompt_paths(self.prompt_versions)
        
        prompts = {}
        for key, path in prompt_paths.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read().strip()
            except FileNotFoundError:
                print(f"Warning: Prompt file not found: {path}")
                print(f"  Strategy: {key}, Version: {self.prompt_versions.get(key, 'unknown')}")
                prompts[key] = ""
        
        print(f"Loaded prompts with versions: {self.prompt_versions}")
        return prompts
    
    def generate_single_strategy(self, original_question: str, strategy: QuestionStrategy, 
                                 query_set_size: int) -> List[str]:
        """
        Generate alternate questions for a single strategy
        
        Args:
            original_question: The original user question
            strategy: Question reformulation strategy
            query_set_size: Number of alternate questions to generate
            
        Returns:
            List of generated questions
        """
        # S1: Parent - just return the original question
        if strategy == QuestionStrategy.PARENT:
            return [original_question]
        
        # For other strategies, use LLM to generate alternates
        if not self.llm:
            return [original_question]
        
        # Get prompt template based on strategy
        prompt_template = None
        if strategy == QuestionStrategy.NEIGHBOR:
            prompt_template = self.prompts.get('neighbor')
        elif strategy == QuestionStrategy.SYNONYMS:
            prompt_template = self.prompts.get('synonyms')
        elif strategy == QuestionStrategy.COMPARATIVE:
            prompt_template = self.prompts.get('comparative')
        elif strategy == QuestionStrategy.HYDE:
            prompt_template = self.prompts.get('hyde')
        elif strategy == QuestionStrategy.QUERY2DOC:
            prompt_template = self.prompts.get('query2doc')

        if not prompt_template:
            return [original_question]

        # Format prompt with variables
        prompt = prompt_template.format(
            query_set_size=query_set_size,
            original_question=original_question
        )

        try:
            system_prompt = self.prompts.get('system', 'You are a helpful assistant.')
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

            # Use unified LLM interface
            response = self.llm.run(full_prompt)
            alternate_text = response.strip()

            # HyDE / Query2Doc: the LLM returns a single hypothetical passage,
            # not a numbered list of question variations. Return the raw passage.
            if strategy == QuestionStrategy.HYDE:
                return [alternate_text] if alternate_text else [original_question]
            if strategy == QuestionStrategy.QUERY2DOC:
                # Canonical Query2Doc concatenates the original query with the pseudo-doc.
                # We do a simple "{query} {passage}" join; for dense retrieval the original
                # paper repeats the query several times, but for BGE simple concat is the
                # standard approximation used in subsequent work.
                if not alternate_text:
                    return [original_question]
                return [f"{original_question} {alternate_text}"]

            # Default: multi-line list of question variations
            alternates = [q.strip() for q in alternate_text.split('\n') if q.strip()]
            # Remove numbering if present (e.g., "1. ", "- ", etc.)
            alternates = [q.lstrip('0123456789.-) ') for q in alternates]
            # Ensure we have the right number of questions
            alternates = alternates[:query_set_size]

            return alternates

        except Exception as e:
            print(f"Error generating alternate questions for {strategy.value}: {e}")
            return [original_question]
    
    def generate_all_strategies(self, original_question: str) -> Dict[str, List[str]]:
        """
        Generate alternate questions using ALL strategies (S1-S4)
        
        Args:
            original_question: The original user question
            
        Returns:
            Dictionary mapping strategy name to list of questions
        """
        print(f"\n{'='*80}")
        print("Generating alternate questions using ALL strategies (S1-S4)")
        print(f"{'='*80}")
        
        all_questions = {}
        
        # S1: Parent (original question)
        print("\n[S1 - Parent]")
        s1_questions = [original_question]
        all_questions['S1'] = s1_questions
        print(f"  Q1: {original_question}")
        
        if not self.openai_client:
            print("\nWarning: No OpenAI client available. Using only original question (Parent strategy).")
            return all_questions
        
        # S2: Neighbor Queries (2 questions)
        print("\n[S2 - Neighbor Queries]")
        s2_questions = self.generate_single_strategy(
            original_question, QuestionStrategy.NEIGHBOR, 2
        )
        all_questions['S2'] = s2_questions
        for i, q in enumerate(s2_questions, 1):
            print(f"  Q{i}: {q}")
        
        # S3: Synonyms (3 questions)
        print("\n[S3 - Synonyms]")
        s3_questions = self.generate_single_strategy(
            original_question, QuestionStrategy.SYNONYMS, 3
        )
        all_questions['S3'] = s3_questions
        for i, q in enumerate(s3_questions, 1):
            print(f"  Q{i}: {q}")
        
        # S4: Comparative (1 question)
        print("\n[S4 - Comparative]")
        s4_questions = self.generate_single_strategy(
            original_question, QuestionStrategy.COMPARATIVE, 1
        )
        all_questions['S4'] = s4_questions
        for i, q in enumerate(s4_questions, 1):
            print(f"  Q{i}: {q}")
        
        total_questions = sum(len(qs) for qs in all_questions.values())
        print(f"\n{'='*80}")
        print(f"Total alternate questions generated: {total_questions}")
        print(f"{'='*80}")
        
        return all_questions
    
    def filter_quality_questions(self, questions: List[str], original_question: str) -> List[str]:
        """
        Filter out low-quality generated questions based on semantic similarity
        
        Args:
            questions: List of generated questions
            original_question: The original user question
            
        Returns:
            Filtered list of questions
        """
        if not ENABLE_QUERY_FILTERING:
            return questions
        
        if len(questions) <= 1:
            return questions
        
        print("\n🔍 Filtering questions by quality...")
        
        # Encode all questions
        original_embedding = self.embedding_model.encode([original_question])
        question_embeddings = self.embedding_model.encode(questions)
        
        # Calculate similarities to original
        similarities = cosine_similarity(question_embeddings, original_embedding).flatten()
        
        filtered_questions = []
        for i, (q, sim) in enumerate(zip(questions, similarities)):
            if MIN_SIMILARITY_TO_ORIGINAL <= sim <= MAX_SIMILARITY_TO_ORIGINAL:
                filtered_questions.append(q)
                print(f"  ✓ Kept: {q[:60]}... (similarity: {sim:.3f})")
            else:
                print(f"  ✗ Filtered: {q[:60]}... (similarity: {sim:.3f})")
        
        if not filtered_questions:
            print("  ⚠️ All questions filtered, keeping original set")
            return questions
        
        print(f"  Kept {len(filtered_questions)}/{len(questions)} questions")
        return filtered_questions
    
    @staticmethod
    def analyze_query_type(query: str) -> str:
        """
        Analyze query type to adapt strategy
        
        Args:
            query: The user question
            
        Returns:
            Query type: 'factual', 'comparative', 'explanatory', 'list', or 'general'
        """
        query_lower = query.lower()
        
        # Factual questions
        if any(word in query_lower for word in ['who', 'what', 'when', 'where', 'which']):
            return 'factual'
        
        # Comparative questions
        if any(word in query_lower for word in ['compare', 'difference', 'better', 'versus', 'vs']):
            return 'comparative'
        
        # Explanatory questions
        if any(word in query_lower for word in ['how', 'why', 'explain', 'describe']):
            return 'explanatory'
        
        # List-based questions
        if any(word in query_lower for word in ['list', 'examples', 'types', 'kinds']):
            return 'list'
        
        return 'general'
