"""
RAG Strategy Configuration
---------------------------
Configure question reformulation and result mixing strategies here.
"""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class QuestionStrategy(Enum):
    """Question reformulation strategies"""
    PARENT = "parent"
    NEIGHBOR = "neighbor"
    SYNONYMS = "synonyms"
    COMPARATIVE = "comparative"
    HYDE = "hyde"            # Replace query with LLM-generated hypothetical passage (Gao et al. 2022)
    QUERY2DOC = "query2doc"  # Concatenate query with LLM-generated pseudo-doc (Wang et al. 2023)


class MixingStrategy(Enum):
    """Result mixing strategies"""
    TOP_K_PER_SET = "top_k_per_set"
    ROUND_ROBIN = "round_robin"
    SCORE_BASED = "score_based"
    DEDUPLICATE = "deduplicate"


# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

# Select which question reformulation strategy to use
# Options: "S1", "S2", "S3", "S4"
SELECTED_STRATEGY = "S3"  # Default: S3 (Synonyms)

# Strategy-specific settings
STRATEGY_SETTINGS = {
    "S1": {
        "strategy": QuestionStrategy.PARENT,
        "query_set_size": 1,
        "description": "Parent - Use original question only"
    },
    "S2": {
        "strategy": QuestionStrategy.NEIGHBOR,
        "query_set_size": 2,
        "description": "Neighbor Queries (Siblings at Same Level)"
    },
    "S3": {
        "strategy": QuestionStrategy.SYNONYMS,
        "query_set_size": 3,
        "description": "Synonyms - Generate synonym-based variations"
    },
    "S4": {
        "strategy": QuestionStrategy.COMPARATIVE,
        "query_set_size": 1,
        "description": "Comparative Neighbor Query"
    }
}


# ============================================================================
# MIXING CONFIGURATION
# ============================================================================

# Select which result mixing strategy to use
# Options: "TOP_K_PER_SET", "ROUND_ROBIN", "SCORE_BASED", "DEDUPLICATE"
SELECTED_MIXING_STRATEGY = "SCORE_BASED"  # Changed to SCORE_BASED for better global ranking

# Mixing parameters
TOP_K_PER_SET = 3  # Reduced from 5 - take fewer per set to reduce noise
MAX_TOTAL_RESULTS = 12  # Reduced from 20 - less context, more focused
REMOVE_DUPLICATES = True  # Whether to remove duplicate documents
SCORE_THRESHOLD = 0.0  # Minimum similarity score (0.0 = no threshold)

# Reranking parameters
ENABLE_RERANKING = True  # Enable cross-encoder reranking
RERANK_TOP_N = 15  # Reduced from 20 - focus on top documents
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model for reranking

# Diversity parameters
ENABLE_MMR = True  # Enable Maximal Marginal Relevance
MMR_LAMBDA = 0.85  # Increased from 0.7 - prioritize relevance more (85% relevance, 15% diversity)

# Query quality filtering
ENABLE_QUERY_FILTERING = True  # Filter low-quality generated questions
MIN_SIMILARITY_TO_ORIGINAL = 0.4  # Increased from 0.3 - stricter filtering
MAX_SIMILARITY_TO_ORIGINAL = 0.90  # Reduced from 0.95 - remove more near-duplicates


# ============================================================================
# VECTOR DB CONFIGURATION
# ============================================================================

# Number of documents to retrieve per query (per strategy)
TOP_K_RETRIEVAL = {
    "S1": 10,  # Parent - most important, retrieve more
    "S2": 7,   # Neighbor queries
    "S3": 5,   # Synonyms
    "S4": 8    # Comparative
}
DEFAULT_TOP_K = 5  # Fallback if strategy not specified

# Embedding model to use
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Collection name for ChromaDB
COLLECTION_NAME = "configurable_rag"


# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Base directory for prompts
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

# Prompt versions for each strategy (can be overridden in config files)
DEFAULT_PROMPT_VERSIONS = {
    "neighbor": os.getenv("NEIGHBOR_PROMPT_VERSION", "v1"),
    "synonyms": os.getenv("SYNONYMS_PROMPT_VERSION", "v1"),
    "comparative": os.getenv("COMPARATIVE_PROMPT_VERSION", "v1"),
    "hyde": os.getenv("HYDE_PROMPT_VERSION", "v1"),
    "query2doc": os.getenv("QUERY2DOC_PROMPT_VERSION", "v1"),
    "system": os.getenv("SYSTEM_PROMPT_VERSION", "v1"),
}

# Build prompt paths based on versions
def _build_prompt_paths(versions: dict = None) -> dict:
    """Build prompt paths for specific versions.
    
    Args:
        versions: Dict mapping strategy name to version (e.g., {"neighbor": "v1", "synonyms": "v2"})
    
    Returns:
        Dict mapping strategy name to file path
    """
    versions = versions or DEFAULT_PROMPT_VERSIONS
    
    return {
        "neighbor": os.path.join(PROMPTS_DIR, "neighbor", f"{versions.get('neighbor', 'v1')}.txt"),
        "synonyms": os.path.join(PROMPTS_DIR, "synonyms", f"{versions.get('synonyms', 'v1')}.txt"),
        "comparative": os.path.join(PROMPTS_DIR, "comparative", f"{versions.get('comparative', 'v1')}.txt"),
        "hyde": os.path.join(PROMPTS_DIR, "hyde", f"{versions.get('hyde', 'v1')}.txt"),
        "query2doc": os.path.join(PROMPTS_DIR, "query2doc", f"{versions.get('query2doc', 'v1')}.txt"),
        "system": os.path.join(PROMPTS_DIR, "system", f"{versions.get('system', 'v1')}.txt"),
    }

# Default prompt paths (using default versions)
PROMPT_PATHS = _build_prompt_paths()

# Prompt metadata for tracking
PROMPT_CONFIG = {
    "versions": DEFAULT_PROMPT_VERSIONS,
    "base_dir": PROMPTS_DIR,
    "paths": PROMPT_PATHS,
}


# ============================================================================
# OPENAI CONFIGURATION
# ============================================================================

# Try to load from openai_config.py first, then fall back to environment variables
try:
    import openai_config as _oai_config
    OPENAI_API_KEY = getattr(_oai_config, 'OPENAI_API_KEY', '') or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
    OPENAI_API_BASE = getattr(_oai_config, 'OPENAI_API_BASE', '') or os.getenv("OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    OPENAI_API_VERSION = getattr(_oai_config, 'OPENAI_API_VERSION', '') or os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    MODEL_NAME = getattr(_oai_config, 'MODEL_NAME', '') or os.getenv("MODEL_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")


# ============================================================================
# COMPARISON MODE CONFIGURATION
# ============================================================================

# Enable comparison mode to compare configured vs non-configured RAG
ENABLE_COMPARISON_MODE = True

# Show detailed output in comparison mode
VERBOSE_COMPARISON = True
