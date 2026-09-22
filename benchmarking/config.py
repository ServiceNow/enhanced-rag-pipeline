"""
Benchmark Configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path

from config.strategy import MixingStrategy


class ChunkingStrategy(Enum):
    """Document chunking strategies for benchmarking"""
    NONE = "none"
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    strategy: ChunkingStrategy = ChunkingStrategy.NONE
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class PromptConfig:
    """Prompt configuration for question generation"""
    versions: Dict[str, str] = field(default_factory=lambda: {
        "neighbor": "v1",
        "synonyms": "v1", 
        "comparative": "v1",
        "system": "v1"
    })
    custom_paths: Optional[Dict[str, str]] = None  # Override specific prompt paths


@dataclass
class DataSourceConfig:
    """Configuration for a single data source (corpus, queries, or qrels)"""
    # For HuggingFace
    dataset: str = ""      # e.g., "BeIR/nfcorpus" or "BeIR/scidocs-generated-queries"
    subset: str = ""       # e.g., "corpus", "queries", or empty for qrels
    split: str = ""        # e.g., "corpus", "queries", "test", "train"
    
    # For local files
    path: str = ""         # e.g., "./data/corpus.jsonl"
    
    def is_huggingface(self) -> bool:
        return bool(self.dataset) and not bool(self.path)
    
    def is_local(self) -> bool:
        return bool(self.path)


@dataclass
class DatasetConfig:
    """Configuration for the complete dataset (corpus + queries + qrels)"""
    name: str = "custom"
    corpus: DataSourceConfig = field(default_factory=DataSourceConfig)
    queries: DataSourceConfig = field(default_factory=DataSourceConfig)
    qrels: DataSourceConfig = field(default_factory=DataSourceConfig)


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name_path: str = "gpt-4.1"
    base_url: str = ""
    max_tokens: int = 1000
    temperature: float = 0.5
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 1024


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks"""
    
    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM settings
    llm_type: str = "gpt"
    llm: LLMConfig = field(default_factory=LLMConfig)
    strict_llm: bool = True
    
    # RAG strategy (S1, S2, S3, S4, or "all")
    strategy: str = "S1"
    
    # Result mixing strategy (for combining multi-query results)
    mixing_strategy: MixingStrategy = MixingStrategy.SCORE_BASED
    
    use_reranking: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_mmr: bool = False
    mmr_lambda: float = 0.7  # Trade-off: 1.0 = relevance only, 0.0 = diversity only
    
    # Chunking
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    
    # Prompts
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    
    # Retrieval settings
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    max_queries: Optional[int] = None
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "ndcg", "map", "recall", "precision", "mrr"
    ])
    
    # Output
    output_dir: str = field(default_factory=lambda: "/enhanced_rag/benchmarking_results" if Path("/enhanced_rag").exists() else "benchmark_results")
    run_suffix: str = ""  # Optional suffix for result directory name (e.g., "bge", "v2")
    save_per_query: bool = False  # Save per-query results
    
    # Reproducibility
    seed: Optional[int] = None  # Random seed for deterministic results
    
    def __post_init__(self):
        if isinstance(self.chunking, dict):
            self.chunking = ChunkingConfig(**self.chunking)
        if isinstance(self.dataset, dict):
            self.dataset = _parse_dataset_config(self.dataset)
        if isinstance(self.mixing_strategy, str):
            self.mixing_strategy = MixingStrategy(self.mixing_strategy)
        if isinstance(self.prompt_config, dict):
            self.prompt_config = PromptConfig(**self.prompt_config)


def _parse_dataset_config(data: dict) -> DatasetConfig:
    """Parse dataset config from dict"""
    def parse_source(source_data) -> DataSourceConfig:
        if isinstance(source_data, str):
            # Simple string - could be HF dataset or local path
            if source_data.startswith("./") or source_data.startswith("/"):
                return DataSourceConfig(path=source_data)
            else:
                return DataSourceConfig(dataset=source_data)
        elif isinstance(source_data, dict):
            return DataSourceConfig(**source_data)
        return DataSourceConfig()
    
    return DatasetConfig(
        name=data.get('name', 'custom'),
        corpus=parse_source(data.get('corpus', {})),
        queries=parse_source(data.get('queries', {})),
        qrels=parse_source(data.get('qrels', {}))
    )
