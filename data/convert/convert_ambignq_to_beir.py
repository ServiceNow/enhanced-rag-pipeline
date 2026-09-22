#!/usr/bin/env python3
"""
Convert AmbigNQ dataset to BEIR format
----------------------------------------
Converts the AmbigNQ JSON file to BEIR format:
- corpus.jsonl: documents from search results
- queries.jsonl: questions
- qrels.tsv: relevance judgments based on viewed_doc_titles

Usage:
# Convert dev set (saves to data/ambignq_beir/)
python convert_ambignq_to_beir.py

# Convert with custom input/output
python convert_ambignq_to_beir.py --input data/ambignq/dev.json --output data/ambignq_beir

# Convert limited samples
python convert_ambignq_to_beir.py --max-samples 1000 --output data/ambignq1000
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Set


def clean_snippet(snippet: str) -> str:
    """Clean HTML tags and formatting from snippet"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', snippet)
    # Replace HTML entities
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&#39;', "'")
    text = text.replace('&middot;', '·')
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_ambignq_to_beir(input_file: str, output_dir: str, max_samples: int = None):
    """Convert AmbigNQ to BEIR format"""
    
    print(f"Loading AmbigNQ from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Limiting to {max_samples} samples")
    
    print(f"Processing {len(data)} questions...")
    
    corpus = {}
    queries = {}
    qrels = {}
    
    # Track document titles to IDs mapping
    title_to_doc_id = {}
    doc_counter = 0
    
    for item in data:
        q_id = str(item['id'])
        question = item['question']
        viewed_titles = set(item.get('viewed_doc_titles', []))
        nq_doc_title = item.get('nq_doc_title', '')
        
        # Add nq_doc_title to viewed titles as it's the primary relevant doc
        if nq_doc_title:
            viewed_titles.add(nq_doc_title)
        
        # Add query
        queries[q_id] = {
            '_id': q_id,
            'text': question
        }
        
        # Process search results from used_queries
        for query_obj in item.get('used_queries', []):
            for result in query_obj.get('results', []):
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                if not title or not snippet:
                    continue
                
                # Create or get document ID based on title
                if title not in title_to_doc_id:
                    doc_id = f"doc_{doc_counter}"
                    title_to_doc_id[title] = doc_id
                    doc_counter += 1
                    
                    # Clean and add to corpus
                    clean_text = clean_snippet(snippet)
                    corpus[doc_id] = {
                        '_id': doc_id,
                        'title': title,
                        'text': clean_text
                    }
                
                # Check if this document is relevant (in viewed_titles)
                if title in viewed_titles:
                    doc_id = title_to_doc_id[title]
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


def main():
    parser = argparse.ArgumentParser(description='Convert AmbigNQ to BEIR format')
    parser.add_argument(
        '--input',
        default='data/ambignq/dev.json',
        help='Input AmbigNQ JSON file (default: data/ambignq/dev.json)'
    )
    parser.add_argument(
        '--output',
        default='data/ambignq_beir',
        help='Output directory for BEIR files (default: data/ambignq_beir)'
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
        print("\nDownload AmbigNQ first from:")
        print("  https://nlp.cs.washington.edu/ambigqa/")
        return 1
    
    convert_ambignq_to_beir(args.input, args.output, args.max_samples)
    return 0


if __name__ == '__main__':
    exit(main())
