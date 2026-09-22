"""
Benchmarking Module for RAG Pipeline
-------------------------------------
Evaluate retrieval performance using BEIR-format datasets.
"""

from .beir_loader import BEIRLoader, Document, Query
from .runner import BenchmarkRunner, BenchmarkResult
from .config import (
    BenchmarkConfig, 
    ChunkingConfig, 
    ChunkingStrategy,
    DatasetConfig,
    DataSourceConfig,
)

# Re-export metrics from evaluation module
from evaluation.metrics import RetrievalMetrics, RetrievalMetricResult

__all__ = [
    'BEIRLoader',
    'Document',
    'Query',
    'BenchmarkRunner',
    'BenchmarkResult', 
    'BenchmarkConfig',
    'ChunkingConfig',
    'ChunkingStrategy',
    'DatasetConfig',
    'DataSourceConfig',
    'RetrievalMetrics',
    'RetrievalMetricResult',
]
