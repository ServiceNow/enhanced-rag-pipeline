#!/usr/bin/env python3
"""
Convert HotpotQA dataset to BEIR format
----------------------------------------
Converts the downloaded HotpotQA JSON file to BEIR format:
- corpus.jsonl: documents from context
- queries.jsonl: questions
- qrels.tsv: relevance judgments based on supporting_facts

Usage:
# Convert 5000 samples (saves to data/hotpotqa5000/)
python convert_hotpotqa_to_beir.py --max-samples 5000 --output data/hotpotqa5000

# Convert full dataset (saves to data/hotpotqa/)
python convert_hotpotqa_to_beir.py

# Convert custom number of samples
python convert_hotpotqa_to_beir.py --max-samples 10000 --output data/hotpotqa10000
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set


def convert_hotpotqa_to_beir(input_file: str, output_dir: str, max_samples: int = None):
    """Convert HotpotQA to BEIR format"""
    
    print(f"Loading HotpotQA from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Limiting to {max_samples} samples")
    
    print(f"Processing {len(data)} questions...")
    
    corpus = {}
    queries = {}
    qrels = {}
    
    for item in data:
        q_id = item['_id']
        question = item['question']
        answer = item.get('answer', '')
        context = item.get('context', [])
        supporting_facts = item.get('supporting_facts', [])
        
        # Add query
        queries[q_id] = {
            '_id': q_id,
            'text': question
        }
        
        # Build set of supporting titles for quick lookup
        supporting_titles = set(sf[0] for sf in supporting_facts)
        
        # Process context documents
        for doc_idx, (title, sentences) in enumerate(context):
            # Create document ID
            doc_id = f"{q_id}_doc{doc_idx}"
            
            # Combine sentences into document text
            if isinstance(sentences, list):
                text = ' '.join(sentences)
            else:
                text = str(sentences)
            
            # Add to corpus
            corpus[doc_id] = {
                '_id': doc_id,
                'title': title,
                'text': text
            }
            
            # Check if this document is relevant (has supporting facts)
            if title in supporting_titles:
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][doc_id] = 1
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save corpus
    corpus_file = output_path / 'corpus.jsonl'
    print(f"\nSaving {len(corpus)} documents to {corpus_file}")
    with open(corpus_file, 'w') as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + '\n')
    
    # Save queries
    queries_file = output_path / 'queries.jsonl'
    print(f"Saving {len(queries)} queries to {queries_file}")
    with open(queries_file, 'w') as f:
        for query in queries.values():
            f.write(json.dumps(query) + '\n')
    
    # Save qrels
    qrels_file = output_path / 'qrels.tsv'
    total_judgments = sum(len(docs) for docs in qrels.values())
    print(f"Saving {total_judgments} relevance judgments to {qrels_file}")
    with open(qrels_file, 'w') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q_id in sorted(qrels.keys()):
            for doc_id, score in sorted(qrels[q_id].items()):
                f.write(f"{q_id}\t{doc_id}\t{score}\n")
    
    print("\n✓ Conversion complete!")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} questions")
    print(f"  Qrels: {total_judgments} relevance judgments")
    print(f"\nFiles saved to: {output_path}")
    print("\nYou can now run:")
    print(f"  python run_benchmark.py benchmark_config_hotpotqa.yaml")


def main():
    parser = argparse.ArgumentParser(description='Convert HotpotQA to BEIR format')
    parser.add_argument(
        '--input',
        default='data/hotpotqa/hotpot_train_v1.1.json',
        help='Input HotpotQA JSON file (default: data/hotpotqa/hotpot_train_v1.1.json)'
    )
    parser.add_argument(
        '--output',
        default='data/hotpotqa',
        help='Output directory for BEIR files (default: data/hotpotqa)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to convert (default: all)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nDownload HotpotQA first:")
        print("  mkdir -p data/hotpotqa")
        print("  curl -L -o data/hotpotqa/hotpot_train_v1.1.json \\")
        print("    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json")
        return 1
    
    convert_hotpotqa_to_beir(args.input, args.output, args.max_samples)
    return 0


if __name__ == '__main__':
    exit(main())
