# Enhanced RAG Pipeline

Reference implementation of the paper's retrieval pipeline and benchmark harness. This repository contains only the implementation needed to run the query-reformulation, retrieval, reranking, and evaluation experiments; it does not include a UI, private datasets, experiment outputs, credentials, or internal infrastructure configuration.

## Included

- Query strategies: S1, S2, S3, S4, HyDE, and Query2Doc
- ChromaDB retrieval, score-based mixing, reranking, and MMR
- BEIR and local BEIR-format dataset loading
- Standard retrieval metrics and significance testing
- Azure OpenAI and optional local vLLM backends
- Public dataset conversion utilities and versioned prompts

## Installation

Python 3.10 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the local vLLM backend on a supported Linux/GPU environment:

```bash
pip install -e '.[vllm]'
```

## Run a benchmark

Copy the public example configuration and select a backend, dataset, and strategy:

```bash
cp benchmarking/configs/benchmark_config.example.yaml benchmark.yaml
python run_benchmark.py benchmark.yaml
```

The example uses public Hugging Face BEIR data, the optional vLLM backend, and writes outputs beneath `./benchmark_results`. Set `strict_llm: true` (the default) to fail rather than silently replace a requested query-reformulation strategy with S1 when LLM initialization fails.

### Azure OpenAI backend

Set `llm_type: gpt` in the benchmark configuration, copy the environment template, and add your local credentials:

```bash
cp .env.template .env
```

Alternatively, export `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION` directly. `AZURE_OPENAI_DEPLOYMENT` is used by the standalone RAG system; benchmark runs take the deployment name from `llm.model_name_path` in the YAML configuration.

Never commit credential files. `.env` files and generated benchmark results are excluded by `.gitignore`.

### vLLM backend

Set `llm_type: vllm`. `llm.model_name_path` may be a Hugging Face model ID, a local path, or a public alias from `core/llm/model_inventory.yaml`.

### OpenAI-compatible local backend

Set `llm_type: openai_compatible` to use a local server such as Docker Model Runner:

```yaml
llm_type: openai_compatible
llm:
  model_name_path: hf.co/HuggingFaceTB/SmolLM2-135M-Instruct
  base_url: http://localhost:12434/engines/v1
  max_tokens: 128
  temperature: 0
```

Docker Model Runner model identifiers are case-sensitive. `max_queries` can limit an end-to-end smoke test without changing the dataset.

## Repository layout

```text
core/           query generation, retrieval, ranking, and LLM adapters
benchmarking/   dataset loader, configuration, and benchmark runner
config/         strategy and prompt configuration
evaluation/     retrieval metrics and significance tests
data/convert/   public dataset conversion utilities
prompts/        versioned prompts used by the strategies
run_benchmark.py
```

## Data and secrets

No datasets or credentials are distributed in this repository. Use public Hugging Face dataset identifiers or local BEIR-format files. Generated results may contain source-document text, so review them before sharing.
