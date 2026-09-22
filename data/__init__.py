"""
Data Module for Enhanced RAG System
------------------------------------
Handles data loading, parsing, and sample datasets
"""

from .loader import DataLoader
from .samples import SAMPLE_DOCUMENTS

__all__ = ['DataLoader', 'SAMPLE_DOCUMENTS']
