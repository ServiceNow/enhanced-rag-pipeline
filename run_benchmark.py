#!/usr/bin/env python3
"""
Run Benchmark Script
--------------------
Usage:
    python run_benchmark.py config.yaml
    python run_benchmark.py benchmark_configs/nfcorpus.yaml
"""

import sys
import yaml
from pathlib import Path
from benchmarking import BenchmarkRunner, BenchmarkConfig
from benchmarking.config import (
    ChunkingConfig, ChunkingStrategy,
    DatasetConfig, DataSourceConfig,
    PromptConfig, LLMConfig
)
from config.strategy import MixingStrategy


def load_config(config_path: str) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse dataset config
    dataset_data = data.get('dataset', {})
    dataset_config = DatasetConfig(
        name=dataset_data.get('name', 'custom'),
        corpus=_parse_source(dataset_data.get('corpus', {})),
        queries=_parse_source(dataset_data.get('queries', {})),
        qrels=_parse_source(dataset_data.get('qrels', {}))
    )
    
    # Parse chunking config
    chunking_data = data.get('chunking', {})
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy(chunking_data.get('strategy', 'none')),
        chunk_size=chunking_data.get('chunk_size', 512),
        chunk_overlap=chunking_data.get('chunk_overlap', 50)
    )
    
    # Parse mixing strategy
    mixing_str = data.get('mixing_strategy', 'score_based')
    mixing_strategy = MixingStrategy(mixing_str)
    
    # Parse prompt config
    prompt_data = data.get('prompt_config', {})
    prompt_config = PromptConfig(
        versions=prompt_data.get('versions', {
            'neighbor': 'v1',
            'synonyms': 'v1',
            'comparative': 'v1',
            'system': 'v1'
        }),
        custom_paths=prompt_data.get('custom_paths', None)
    )
    
    # Parse LLM config
    llm_data = data.get('llm', {})
    llm_config = LLMConfig(
        model_name_path=llm_data.get('model_name_path', 'gpt-4.1'),
        base_url=llm_data.get('base_url', ''),
        max_tokens=llm_data.get('max_tokens', 1000),
        temperature=llm_data.get('temperature', 0.5),
        gpu_memory_utilization=llm_data.get('gpu_memory_utilization', 0.85),
        max_model_len=llm_data.get('max_model_len', 1024)
    )
    
    # Build main config
    config = BenchmarkConfig(
        dataset=dataset_config,
        embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2'),
        llm_type=data.get('llm_type', 'gpt'),
        llm=llm_config,
        strict_llm=data.get('strict_llm', True),
        strategy=data.get('strategy', 'S1'),
        mixing_strategy=mixing_strategy,
        use_reranking=data.get('use_reranking', False),
        rerank_model=data.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        use_mmr=data.get('use_mmr', False),
        mmr_lambda=data.get('mmr_lambda', 0.7),
        chunking=chunking_config,
        prompt_config=prompt_config,
        top_k_values=data.get('top_k_values', [1, 3, 5, 10, 20, 100]),
        max_queries=data.get('max_queries', None),
        metrics=data.get('metrics', ['ndcg', 'map', 'recall', 'precision', 'mrr']),
        output_dir=data.get('output_dir', 'benchmark_results'),
        run_suffix=data.get('run_suffix', ''),
        save_per_query=data.get('save_per_query', False),
        seed=data.get('seed', None)
    )
    
    return config


def _parse_source(source_data) -> DataSourceConfig:
    """Parse a data source config from YAML."""
    if isinstance(source_data, str):
        # Simple string - HF dataset or local path
        if source_data.startswith('./') or source_data.startswith('/'):
            return DataSourceConfig(path=source_data)
        else:
            return DataSourceConfig(dataset=source_data)
    elif isinstance(source_data, dict):
        return DataSourceConfig(
            dataset=source_data.get('dataset', ''),
            subset=source_data.get('subset', ''),
            split=source_data.get('split', ''),
            path=source_data.get('path', '')
        )
    return DataSourceConfig()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a retrieval benchmark from a YAML config.")
    parser.add_argument("config", help="Path to the benchmark YAML config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in the YAML (used for multi-seed reproducibility runs)")
    parser.add_argument("--run-suffix", default=None,
                        help="Override run_suffix in the YAML (appended to output dir name)")
    args = parser.parse_args()

    config_path = args.config

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    if args.seed is not None:
        config.seed = args.seed
    if args.run_suffix is not None:
        config.run_suffix = args.run_suffix

    runner = BenchmarkRunner(config, seed=config.seed)
    runner.config_path = config_path
    result = runner.run()

    return result


if __name__ == '__main__':
    main()
