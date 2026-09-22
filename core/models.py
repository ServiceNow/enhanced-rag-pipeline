"""
Data models and configurations for RAG strategies
"""

from dataclasses import dataclass
from typing import Optional

# Import enums from config (single source of truth)
from config.strategy import QuestionStrategy, MixingStrategy


@dataclass
class StrategyConfig:
    """Configuration for question reformulation strategy"""
    strategy: QuestionStrategy
    query_set_size: int  # Number of alternate questions to generate
    description: str
    
    def __post_init__(self):
        """Validate configuration"""
        if self.query_set_size < 1:
            raise ValueError("query_set_size must be at least 1")


@dataclass
class MixingConfig:
    """Configuration for result mixing and sorting"""
    strategy: MixingStrategy
    top_k_per_set: int = 5  # For TOP_K_PER_SET strategy
    max_total_results: int = 15  # Maximum total results to keep
    remove_duplicates: bool = True
    score_threshold: float = 0.0  # Minimum similarity score to include
    
    def __post_init__(self):
        """Validate configuration"""
        if self.top_k_per_set < 1:
            raise ValueError("top_k_per_set must be at least 1")
        if self.max_total_results < 1:
            raise ValueError("max_total_results must be at least 1")


# Predefined strategy configurations based on the table
STRATEGY_CONFIGS = {
    "S1": StrategyConfig(
        strategy=QuestionStrategy.PARENT,
        query_set_size=1,
        description="Parent - Use original question only"
    ),
    "S2": StrategyConfig(
        strategy=QuestionStrategy.NEIGHBOR,
        query_set_size=2,
        description="Neighbor Queries (Siblings at Same Level)"
    ),
    "S3": StrategyConfig(
        strategy=QuestionStrategy.SYNONYMS,
        query_set_size=3,
        description="Synonyms - Generate synonym-based variations"
    ),
    "S4": StrategyConfig(
        strategy=QuestionStrategy.COMPARATIVE,
        query_set_size=1,
        description="Comparative Neighbor Query"
    ),
    "HyDE": StrategyConfig(
        strategy=QuestionStrategy.HYDE,
        query_set_size=1,
        description="HyDE - replace query with LLM-generated hypothetical passage (Gao et al. 2022)"
    ),
    "Query2Doc": StrategyConfig(
        strategy=QuestionStrategy.QUERY2DOC,
        query_set_size=1,
        description="Query2Doc - concatenate query with LLM-generated pseudo-doc (Wang et al. 2023)"
    ),
}
