"""
BEIR Format Data Loader
-----------------------
Load corpus, queries, and qrels from various sources.

Each component (corpus, queries, qrels) can come from:
1. HuggingFace dataset (dataset + subset + split)
2. Local file (path to .jsonl or .tsv)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .config import DatasetConfig, DataSourceConfig


@dataclass
class Document:
    """A document in the corpus"""
    doc_id: str
    text: str
    title: str = ""


@dataclass  
class Query:
    """A query"""
    query_id: str
    text: str


class BEIRLoader:
    """Load BEIR-format datasets from local files or HuggingFace"""
    
    def __init__(self, dataset_config: "DatasetConfig"):
        """
        Initialize loader with dataset configuration.
        
        Args:
            dataset_config: DatasetConfig specifying sources for corpus, queries, qrels
        """
        self.config = dataset_config
        self.corpus: Dict[str, Document] = {}
        self.queries: Dict[str, Query] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
    
    def load(self) -> Tuple[Dict[str, Document], Dict[str, Query], Dict[str, Dict[str, int]]]:
        """
        Load corpus, queries, and qrels from configured sources.
        
        Returns:
            Tuple of (corpus, queries, qrels)
        """
        print(f"Loading dataset: {self.config.name}")
        
        # Load each component from its source
        self._load_corpus()
        self._load_queries()
        self._load_qrels()
        
        print(f"  Corpus: {len(self.corpus)} documents")
        print(f"  Queries: {len(self.queries)} queries")
        print(f"  Qrels: {sum(len(v) for v in self.qrels.values())} judgments")
        
        return self.corpus, self.queries, self.qrels
    
    # =========================================================================
    # Corpus Loading
    # =========================================================================
    
    def _load_corpus(self) -> None:
        """Load corpus from configured source"""
        source = self.config.corpus
        
        if source.is_local():
            self._load_corpus_local(source.path)
        elif source.is_huggingface():
            self._load_corpus_hf(source.dataset, source.subset, source.split)
        else:
            raise ValueError("Corpus source not configured")
    
    def _load_corpus_local(self, path: str) -> None:
        """Load corpus from local JSONL file"""
        print(f"  Loading corpus from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_id = str(obj.get('_id', obj.get('id', '')))
                title = obj.get('title', '')
                text = obj.get('text', '')
                full_text = f"{title}\n{text}" if title else text
                self.corpus[doc_id] = Document(doc_id=doc_id, text=full_text.strip(), title=title)
    
    def _load_corpus_hf(self, dataset: str, subset: str = "", split: str = "") -> None:
        """Load corpus from HuggingFace"""
        from datasets import load_dataset
        
        subset = subset or "corpus"
        split = split or "corpus"
        
        print(f"  Loading corpus from HF: {dataset} [{subset}] ({split})")
        
        # Try multiple loading strategies for compatibility
        ds = None
        errors = []
        
        # Strategy 1: Try with subset and split
        try:
            ds = load_dataset(dataset, subset, split=split)
        except Exception as e:
            errors.append(f"With subset '{subset}': {str(e)[:100]}")
        
        # Strategy 2: Try without subset
        if ds is None:
            try:
                print(f"  Trying without subset...")
                ds = load_dataset(dataset, split=split)
            except Exception as e:
                errors.append(f"Without subset: {str(e)[:100]}")
        
        # Strategy 3: Try with name parameter instead of positional
        if ds is None:
            try:
                print(f"  Trying with name parameter...")
                ds = load_dataset(dataset, name=subset, split=split)
            except Exception as e:
                errors.append(f"With name parameter: {str(e)[:100]}")
        
        # Strategy 4: Load all splits and select the one we want
        if ds is None:
            try:
                print(f"  Trying to load all splits...")
                all_ds = load_dataset(dataset, subset)
                if split in all_ds:
                    ds = all_ds[split]
                else:
                    # Try to find a matching split
                    available_splits = list(all_ds.keys())
                    print(f"  Available splits: {available_splits}")
                    if available_splits:
                        ds = all_ds[available_splits[0]]
                        print(f"  Using split: {available_splits[0]}")
            except Exception as e:
                errors.append(f"Loading all splits: {str(e)[:100]}")
        
        if ds is None:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(
                f"Failed to load corpus from {dataset}.\n"
                f"Attempted strategies:\n{error_msg}\n\n"
                f"This dataset may use an unsupported loading script or may not be available.\n"
                f"Try downloading the dataset locally and using the 'path' option instead."
            )
        
        for item in ds:
            doc_id = str(item.get('_id', item.get('id', '')))
            title = item.get('title', '')
            text = item.get('text', '')
            full_text = f"{title}\n{text}" if title else text
            self.corpus[doc_id] = Document(doc_id=doc_id, text=full_text.strip(), title=title)
    
    # =========================================================================
    # Queries Loading
    # =========================================================================
    
    def _load_queries(self) -> None:
        """Load queries from configured source"""
        source = self.config.queries
        
        if source.is_local():
            self._load_queries_local(source.path)
        elif source.is_huggingface():
            self._load_queries_hf(source.dataset, source.subset, source.split)
        else:
            raise ValueError("Queries source not configured")
    
    def _load_queries_local(self, path: str) -> None:
        """Load queries from local JSONL file"""
        print(f"  Loading queries from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                query_id = str(obj.get('_id', obj.get('id', '')))
                text = obj.get('text', obj.get('query', ''))
                self.queries[query_id] = Query(query_id=query_id, text=text.strip())
    
    def _load_queries_hf(self, dataset: str, subset: str = "", split: str = "") -> None:
        """Load queries from HuggingFace"""
        from datasets import load_dataset
        
        subset = subset or "queries"
        split = split or "queries"
        
        print(f"  Loading queries from HF: {dataset} [{subset}] ({split})")
        
        ds = None
        errors = []
        
        # Try multiple loading strategies
        try:
            ds = load_dataset(dataset, subset, split=split)
        except Exception as e:
            errors.append(f"With subset '{subset}': {str(e)[:100]}")
        
        if ds is None:
            try:
                print(f"  Trying without subset...")
                ds = load_dataset(dataset, split=split)
            except Exception as e:
                errors.append(f"Without subset: {str(e)[:100]}")
        
        if ds is None:
            try:
                print(f"  Trying with name parameter...")
                ds = load_dataset(dataset, name=subset, split=split)
            except Exception as e:
                errors.append(f"With name parameter: {str(e)[:100]}")
        
        if ds is None:
            try:
                print(f"  Trying to load all splits...")
                all_ds = load_dataset(dataset, subset)
                if split in all_ds:
                    ds = all_ds[split]
                else:
                    available_splits = list(all_ds.keys())
                    print(f"  Available splits: {available_splits}")
                    if available_splits:
                        ds = all_ds[available_splits[0]]
                        print(f"  Using split: {available_splits[0]}")
            except Exception as e:
                errors.append(f"Loading all splits: {str(e)[:100]}")
        
        if ds is None:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(
                f"Failed to load queries from {dataset}.\n"
                f"Attempted strategies:\n{error_msg}\n\n"
                f"Try downloading the dataset locally and using the 'path' option instead."
            )
        
        for item in ds:
            query_id = str(item.get('_id', item.get('id', '')))
            text = item.get('text', item.get('query', ''))
            self.queries[query_id] = Query(query_id=query_id, text=text.strip())
    
    # =========================================================================
    # Qrels Loading
    # =========================================================================
    
    def _load_qrels(self) -> None:
        """Load qrels from configured source"""
        source = self.config.qrels
        
        if source.is_local():
            self._load_qrels_local(source.path)
        elif source.is_huggingface():
            self._load_qrels_hf(source.dataset, source.split)
        else:
            raise ValueError("Qrels source not configured")
    
    def _load_qrels_local(self, path: str) -> None:
        """Load qrels from local TSV file"""
        print(f"  Loading qrels from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Skip header if present
            first_row = next(reader, None)
            if first_row and not first_row[0].replace('-', '').isdigit():
                pass  # Was header
            elif first_row:
                self._add_qrel_row(first_row)
            
            for row in reader:
                self._add_qrel_row(row)
    
    def _load_qrels_hf(self, dataset: str, split: str = "") -> None:
        """Load qrels from HuggingFace"""
        from datasets import load_dataset
        
        split = split or "test"
        
        print(f"  Loading qrels from HF: {dataset} ({split})")
        
        ds = None
        errors = []
        
        # Try multiple loading strategies
        try:
            ds = load_dataset(dataset, split=split)
        except Exception as e:
            errors.append(f"With split '{split}': {str(e)[:100]}")
        
        if ds is None:
            try:
                print(f"  Trying to load all splits...")
                all_ds = load_dataset(dataset)
                if split in all_ds:
                    ds = all_ds[split]
                else:
                    available_splits = list(all_ds.keys())
                    print(f"  Available splits: {available_splits}")
                    if available_splits:
                        ds = all_ds[available_splits[0]]
                        print(f"  Using split: {available_splits[0]}")
            except Exception as e:
                errors.append(f"Loading all splits: {str(e)[:100]}")
        
        if ds is None:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(
                f"Failed to load qrels from {dataset}.\n"
                f"Attempted strategies:\n{error_msg}\n\n"
                f"Try downloading the dataset locally and using the 'path' option instead."
            )
        
        for item in ds:
            query_id = str(item.get('query-id', item.get('query_id', '')))
            doc_id = str(item.get('corpus-id', item.get('corpus_id', item.get('doc_id', ''))))
            score = int(item.get('score', 1))
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {}
            self.qrels[query_id][doc_id] = score
    
    def _add_qrel_row(self, row: List[str]) -> None:
        """Add a single qrel row"""
        if len(row) < 2:
            return
        
        query_id = str(row[0])
        doc_id = str(row[1]) if len(row) == 3 else str(row[2])
        score = int(row[-1]) if len(row) >= 3 else 1
        
        if query_id not in self.qrels:
            self.qrels[query_id] = {}
        self.qrels[query_id][doc_id] = score
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def get_relevant_docs(self, query_id: str, min_relevance: int = 1) -> List[str]:
        """Get list of relevant document IDs for a query"""
        if query_id not in self.qrels:
            return []
        return [
            doc_id for doc_id, score in self.qrels[query_id].items() 
            if score >= min_relevance
        ]
    
    def get_documents_text(self) -> List[str]:
        """Get all document texts as a list"""
        return [doc.text for doc in self.corpus.values()]
    
    def get_document_ids(self) -> List[str]:
        """Get all document IDs as a list"""
        return list(self.corpus.keys())
