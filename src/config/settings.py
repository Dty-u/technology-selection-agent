"""Application settings loaded from environment variables and YAML."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parents[1]

load_dotenv(_PROJECT_ROOT / ".env", override=True)

with (_CONFIG_DIR / "config.yaml").open(encoding="utf-8") as config_file:
    _CONFIG = yaml.safe_load(config_file)


ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

LLM_MODEL = _CONFIG["llm"]["model"]
LLM_TEMPERATURE = _CONFIG["llm"]["temperature"]
LLM_BASE_URL = _CONFIG["llm"]["base_url"]
LLM_REQUEST_TIMEOUT_SECONDS = _CONFIG["llm"]["request_timeout_seconds"]
LLM_MAX_RETRIES = _CONFIG["llm"]["max_retries"]
UTILITY_LLM_TIMEOUT_SECONDS = _CONFIG["llm"]["utility_timeout_seconds"]
UTILITY_LLM_MAX_RETRIES = _CONFIG["llm"]["utility_max_retries"]

ZHIPU_EMBEDDING_MODEL = _CONFIG["embedding"]["zhipu_model"]
ZHIPU_EMBEDDING_BASE_URL = _CONFIG["embedding"]["zhipu_base_url"]
BGE_MODEL_NAME = _CONFIG["embedding"]["bge_model"]
BGE_DEVICE = _CONFIG["embedding"]["device"]

MILVUS_URI = _CONFIG["milvus"]["uri"]
COLLECTION_NAME = _CONFIG["milvus"]["collection_name"]

NEO4J_URI = _CONFIG["neo4j"]["uri"]
NEO4J_USERNAME = _CONFIG["neo4j"]["username"]

RETRIEVER_K = _CONFIG["retriever"]["k"]
RETRIEVER_FETCH_K = _CONFIG["retriever"]["fetch_k"]
RETRIEVER_HNSW_EF = _CONFIG["retriever"]["hnsw_ef"]

GRAPH_MAX_HOPS = _CONFIG["graph_retriever"]["max_hops"]
GRAPH_SEED_LIMIT = _CONFIG["graph_retriever"]["seed_limit"]
GRAPH_PATH_LIMIT = _CONFIG["graph_retriever"]["path_limit"]
GRAPH_MAX_DOCUMENTS = _CONFIG["graph_retriever"]["max_documents"]

RERANKER_ENABLED = bool(_CONFIG["reranker"]["enabled"])
RERANKER_MODEL_NAME = _CONFIG["reranker"]["model"]
RERANKER_BASE_URL = _CONFIG["reranker"]["base_url"]
RERANKER_TOP_N = _CONFIG["reranker"]["top_n"]
RERANKER_TIMEOUT_SECONDS = _CONFIG["reranker"]["request_timeout_seconds"]
RERANKER_MAX_RETRIES = _CONFIG["reranker"]["max_retries"]

RELEVANCE_MAX_DOCUMENTS = _CONFIG["relevance"]["max_documents"]
RELEVANCE_DOCUMENT_MAX_CHARACTERS = _CONFIG["relevance"][
    "document_max_characters"
]

WEB_SEARCH_MAX_RESULTS = _CONFIG["web_search"]["max_results"]
WEB_SEARCH_ENGINE = _CONFIG["web_search"]["search_engine"]
WEB_SEARCH_MAX_ATTEMPTS = _CONFIG["web_search"]["max_attempts"]
WEB_SEARCH_TIMEOUT_SECONDS = _CONFIG["web_search"]["request_timeout_seconds"]

DOCUMENT_CHUNK_LENGTH_THRESHOLD = _CONFIG["document"]["chunk_length_threshold"]
REPORT_OUTPUT_DIR = _PROJECT_ROOT / _CONFIG["report"]["output_dir"]
CHECKPOINT_BACKEND = os.getenv(
    "CHECKPOINT_BACKEND",
    _CONFIG["checkpoint"]["backend"],
).strip().casefold()
CHECKPOINT_DB_PATH = (
    _PROJECT_ROOT / _CONFIG["checkpoint"]["sqlite_archive_path"]
)
CHECKPOINT_SETUP_ON_START = bool(_CONFIG["checkpoint"]["setup_on_start"])
MYSQL_CHECKPOINT_URI = os.getenv("MYSQL_CHECKPOINT_URI", "").strip()
MYSQL_MEMORY_URI = os.getenv("MYSQL_MEMORY_URI", "").strip()

MEMORY_ENABLED = bool(_CONFIG["memory"]["enabled"])
MEMORY_COLLECTION_NAME = _CONFIG["memory"]["milvus_collection_name"]
SHORT_TERM_TURN_LIMIT = _CONFIG["memory"]["short_term_turn_limit"]
MEMORY_EXACT_RECALL_LIMIT = _CONFIG["memory"]["exact_recall_limit"]
MEMORY_SEMANTIC_RECALL_LIMIT = _CONFIG["memory"]["semantic_recall_limit"]
MEMORY_HISTORICAL_CASE_LIMIT = _CONFIG["memory"]["historical_case_limit"]
MEMORY_OUTBOX_MAX_ATTEMPTS = _CONFIG["memory"]["outbox_max_attempts"]

MAX_TRANSFORM_COUNT = _CONFIG["workflow"]["max_transform_count"]
MAX_GENERATION_ATTEMPTS = _CONFIG["workflow"]["max_generation_attempts"]

MAX_SELECTION_CANDIDATES = _CONFIG["selection"]["max_candidates"]
MAX_EVIDENCE_ROUNDS = _CONFIG["selection"]["max_evidence_rounds"]
MAX_EVIDENCE_TASK_CONCURRENCY = _CONFIG["selection"][
    "max_evidence_task_concurrency"
]
SELECTION_WEB_SEARCH_MAX_RESULTS = _CONFIG["selection"][
    "web_search_max_results"
]
SELECTION_WEB_SEARCH_ENGINE = _CONFIG["selection"]["web_search_engine"]
MAX_REVIEW_ROUNDS = _CONFIG["selection"]["max_review_rounds"]
SELECTION_STRICT_CACHE_ENABLED = _CONFIG["selection"][
    "strict_cache_enabled"
]
SELECTION_STRICT_CACHE_MAX_ENTRIES = _CONFIG["selection"][
    "strict_cache_max_entries"
]
SELECTION_STRICT_CACHE_TTL_SECONDS = _CONFIG["selection"][
    "strict_cache_ttl_seconds"
]
SELECTION_RETRIEVAL_CACHE_TTL_SECONDS = _CONFIG["selection"][
    "retrieval_cache_ttl_seconds"
]
MAX_EVIDENCE_CARD_CHARACTERS = _CONFIG["selection"][
    "evidence_card_max_characters"
]
EVIDENCE_ASSESSMENT_BATCH_SIZE = _CONFIG["selection"][
    "evidence_assessment_batch_size"
]
MAX_EVIDENCE_ASSESSMENT_CONCURRENCY = _CONFIG["selection"][
    "max_evidence_assessment_concurrency"
]
EVIDENCE_ASSESSMENT_SINGLE_BATCH_LIMIT = _CONFIG["selection"][
    "evidence_assessment_single_batch_limit"
]
DECISION_EVIDENCE_CARD_MAX_CHARACTERS = _CONFIG["selection"][
    "decision_evidence_card_max_characters"
]
MAX_DECISION_EVIDENCE_CARDS_PER_CRITERION = _CONFIG["selection"][
    "max_decision_evidence_cards_per_criterion"
]
SELECTION_RELEVANCE_DOCUMENT_MAX_CHARACTERS = _CONFIG["selection"][
    "relevance_document_max_characters"
]

WARMUP_ON_START = bool(_CONFIG["performance"]["warmup_on_start"])
RESOURCE_FAILURE_COOLDOWN_SECONDS = _CONFIG["performance"][
    "resource_failure_cooldown_seconds"
]
