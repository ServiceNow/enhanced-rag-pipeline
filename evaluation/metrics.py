"""
RAG Evaluation Metrics
----------------------
Metrics for evaluating RAG systems:
- Answer quality metrics (LLM-based): relevance, completeness, accuracy, etc.
- Retrieval metrics (IR-based): NDCG, MAP, Recall, Precision, MRR, Hit@k
"""

import numpy as np
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from openai import AzureOpenAI


@dataclass
class EvaluationScore:
    """Scores for a single answer"""
    relevance: float  # 0-10: How relevant is the answer to the question
    completeness: float  # 0-10: How complete/comprehensive is the answer
    accuracy: float  # 0-10: How accurate is the answer based on context
    coherence: float  # 0-10: How well-structured and coherent is the answer
    specificity: float  # 0-10: How specific and detailed is the answer
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        return (
            self.relevance * 0.30 +
            self.completeness * 0.25 +
            self.accuracy * 0.25 +
            self.coherence * 0.10 +
            self.specificity * 0.10
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'relevance': self.relevance,
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'coherence': self.coherence,
            'specificity': self.specificity,
            'overall': self.overall_score
        }


class RAGEvaluator:
    """Evaluates RAG answers using LLM-based scoring"""
    
    def __init__(self, openai_client: AzureOpenAI, deployment_name: str):
        """Initialize evaluator with OpenAI client"""
        self.openai_client = openai_client
        self.deployment_name = deployment_name
    
    def evaluate_answer(self, 
                       question: str, 
                       answer: str, 
                       context_docs: List[str]) -> EvaluationScore:
        """
        Evaluate a single answer using LLM-based scoring
        
        Args:
            question: The original question
            answer: The generated answer
            context_docs: The context documents used
            
        Returns:
            EvaluationScore with individual and overall scores
        """
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
        
        evaluation_prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the following answer based on the question and provided context documents.

Question: {question}

Context Documents:
{context}

Generated Answer: {answer}

Please evaluate the answer on the following criteria (score each from 0-10):

1. RELEVANCE (0-10): How well does the answer address the question?
   - 0-3: Not relevant, doesn't answer the question
   - 4-6: Partially relevant, addresses some aspects
   - 7-8: Mostly relevant, addresses main points
   - 9-10: Highly relevant, directly and fully answers the question

2. COMPLETENESS (0-10): How comprehensive is the answer?
   - 0-3: Incomplete, missing major information
   - 4-6: Covers basic points but lacks depth
   - 7-8: Covers most important aspects
   - 9-10: Comprehensive, covers all relevant aspects

3. ACCURACY (0-10): How accurate is the answer based on the context?
   - 0-3: Contains significant inaccuracies or contradictions
   - 4-6: Mostly accurate but some minor issues
   - 7-8: Accurate with no major errors
   - 9-10: Completely accurate and faithful to context

4. COHERENCE (0-10): How well-structured and easy to understand is the answer?
   - 0-3: Confusing, poorly structured
   - 4-6: Understandable but could be clearer
   - 7-8: Clear and well-organized
   - 9-10: Exceptionally clear and well-structured

5. SPECIFICITY (0-10): How specific and detailed is the answer?
   - 0-3: Very vague and generic
   - 4-6: Some specific details but mostly general
   - 7-8: Good level of specific details
   - 9-10: Highly specific with concrete examples and details

Provide your evaluation in the following format (one score per line):
RELEVANCE: [score]
COMPLETENESS: [score]
ACCURACY: [score]
COHERENCE: [score]
SPECIFICITY: [score]

Only provide the scores, no additional explanation."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert RAG system evaluator. Provide objective numerical scores."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent scoring
            )
            
            # Parse the response
            evaluation_text = response.choices[0].message.content.strip()
            scores = self._parse_scores(evaluation_text)
            
            return EvaluationScore(**scores)
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Return default scores if evaluation fails
            return EvaluationScore(
                relevance=5.0,
                completeness=5.0,
                accuracy=5.0,
                coherence=5.0,
                specificity=5.0
            )
    
    def _parse_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse scores from LLM response"""
        scores = {
            'relevance': 5.0,
            'completeness': 5.0,
            'accuracy': 5.0,
            'coherence': 5.0,
            'specificity': 5.0
        }
        
        lines = evaluation_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Extract numeric value
                try:
                    # Handle cases like "8.5" or "8/10" or "8"
                    if '/' in value:
                        value = value.split('/')[0]
                    score = float(value)
                    
                    if key in scores:
                        scores[key] = min(10.0, max(0.0, score))  # Clamp between 0-10
                except ValueError:
                    continue
        
        return scores
    
    def compare_answers(self,
                       question: str,
                       simple_answer: str,
                       simple_context: List[str],
                       configured_answer: str,
                       configured_context: List[str]) -> Dict[str, any]:
        """
        Compare two answers and return detailed comparison
        
        Returns:
            Dictionary with scores and comparison analysis
        """
        print("\n🔍 Evaluating Simple RAG answer...")
        simple_score = self.evaluate_answer(question, simple_answer, simple_context)
        
        print("🔍 Evaluating Configured RAG answer...")
        configured_score = self.evaluate_answer(question, configured_answer, configured_context)
        
        # Calculate improvements
        improvements = {
            'relevance': configured_score.relevance - simple_score.relevance,
            'completeness': configured_score.completeness - simple_score.completeness,
            'accuracy': configured_score.accuracy - simple_score.accuracy,
            'coherence': configured_score.coherence - simple_score.coherence,
            'specificity': configured_score.specificity - simple_score.specificity,
            'overall': configured_score.overall_score - simple_score.overall_score
        }
        
        return {
            'simple_score': simple_score,
            'configured_score': configured_score,
            'improvements': improvements,
            'winner': 'configured' if configured_score.overall_score > simple_score.overall_score else 'simple'
        }


def display_evaluation_results(comparison: Dict[str, any], strategy_name: str):
    """Display evaluation results in a formatted way"""
    simple_score = comparison['simple_score']
    configured_score = comparison['configured_score']
    improvements = comparison['improvements']
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION METRICS".center(80))
    print("=" * 80)
    
    # Create comparison table
    metrics = ['relevance', 'completeness', 'accuracy', 'coherence', 'specificity', 'overall']
    metric_names = ['Relevance', 'Completeness', 'Accuracy', 'Coherence', 'Specificity', 'Overall Score']
    
    print(f"\n{'Metric':<20} {'Simple RAG':<15} {f'Configured ({strategy_name})':<20} {'Improvement':<15}")
    print("─" * 80)
    
    for metric, name in zip(metrics, metric_names):
        if metric == 'overall':
            simple_val = simple_score.overall_score
            configured_val = configured_score.overall_score
        else:
            simple_val = getattr(simple_score, metric)
            configured_val = getattr(configured_score, metric)
        
        improvement = improvements[metric]
        improvement_str = f"{improvement:+.2f}"
        
        # Add color indicators
        if improvement > 0:
            indicator = "📈"
        elif improvement < 0:
            indicator = "📉"
        else:
            indicator = "➡️"
        
        print(f"{name:<20} {simple_val:<15.2f} {configured_val:<20.2f} {indicator} {improvement_str:<12}")
    
    print("─" * 80)
    
    # Winner announcement
    winner = comparison['winner']
    if winner == 'configured':
        print(f"\n🏆 Winner: Configured RAG ({strategy_name})")
        print(f"   Overall improvement: +{improvements['overall']:.2f} points")
    else:
        print(f"\n🏆 Winner: Simple RAG")
        print(f"   Overall difference: {improvements['overall']:.2f} points")
    
    # Key insights
    print("\n💡 Key Insights:")
    
    # Find best improvements
    best_improvement = max(improvements.items(), key=lambda x: x[1] if x[0] != 'overall' else -999)
    worst_improvement = min(improvements.items(), key=lambda x: x[1] if x[0] != 'overall' else 999)
    
    if best_improvement[1] > 0:
        print(f"   • Strongest improvement: {best_improvement[0].capitalize()} (+{best_improvement[1]:.2f})")
    
    if worst_improvement[1] < 0:
        print(f"   • Area for improvement: {worst_improvement[0].capitalize()} ({worst_improvement[1]:.2f})")
    
    if improvements['overall'] > 1.0:
        print(f"   • Configured RAG shows significant improvement over Simple RAG")
    elif improvements['overall'] > 0:
        print(f"   • Configured RAG shows slight improvement over Simple RAG")
    else:
        print(f"   • Both approaches performed similarly")
    
    print("\n" + "=" * 80)


# =============================================================================
# RETRIEVAL METRICS (IR-based)
# =============================================================================

@dataclass
class RetrievalMetricResult:
    """Results for a single retrieval metric across all queries"""
    name: str
    k: int
    mean: float
    std: float
    per_query: Dict[str, float] = field(default_factory=dict)


class RetrievalMetrics:
    """Compute standard IR retrieval metrics"""
    
    @staticmethod
    def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Hit@k: 1 if at least one relevant doc in top-k, else 0."""
        if not relevant or k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        for doc in retrieved_at_k:
            if doc in relevant:
                return 1.0
        return 0.0
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@k: proportion of top-k retrieved docs that are relevant."""
        if k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@k: proportion of relevant docs found in top-k."""
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_at_k & relevant)
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Average Precision@k: average of precision values at each relevant doc."""
        if not relevant:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_at_k, 1):
            if doc in relevant:
                relevant_count += 1
                precisions.append(relevant_count / i)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / min(len(relevant), k)
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Reciprocal Rank: 1/rank of first relevant document."""
        retrieved_at_k = retrieved[:k]
        
        for i, doc in enumerate(retrieved_at_k, 1):
            if doc in relevant:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
        """NDCG@k: Normalized Discounted Cumulative Gain."""
        if not qrels:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_at_k, 1):
            rel = qrels.get(doc, 0)
            dcg += (2**rel - 1) / np.log2(i + 1)
        
        # Ideal DCG
        ideal_rels = sorted(qrels.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, 1):
            idcg += (2**rel - 1) / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @classmethod
    def compute_all(cls, 
                   retrieved: List[str], 
                   relevant: Set[str],
                   qrels: Dict[str, int],
                   k: int) -> Dict[str, float]:
        """Compute all metrics for a single query."""
        return {
            'hit': cls.hit_at_k(retrieved, relevant, k),
            'precision': cls.precision_at_k(retrieved, relevant, k),
            'recall': cls.recall_at_k(retrieved, relevant, k),
            'map': cls.average_precision(retrieved, relevant, k),
            'mrr': cls.reciprocal_rank(retrieved, relevant, k),
            'ndcg': cls.ndcg_at_k(retrieved, qrels, k),
        }
    
    @classmethod
    def aggregate(cls,
                 all_results: Dict[str, Dict[str, float]],
                 metric_name: str,
                 k: int) -> RetrievalMetricResult:
        """Aggregate per-query results into mean/std."""
        values = [
            results.get(metric_name, 0.0) 
            for results in all_results.values()
        ]
        
        per_query = {
            qid: results.get(metric_name, 0.0)
            for qid, results in all_results.items()
        }
        
        return RetrievalMetricResult(
            name=metric_name,
            k=k,
            mean=float(np.mean(values)) if values else 0.0,
            std=float(np.std(values)) if values else 0.0,
            per_query=per_query
        )
