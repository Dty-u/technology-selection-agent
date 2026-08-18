"""Production composition root for checkpointed, memory-aware workflows."""

from contextlib import contextmanager
from typing import Any, Iterator

from src.config.settings import (
    MYSQL_CHECKPOINT_URI,
    MYSQL_MEMORY_URI,
    WARMUP_ON_START,
)
from src.config.bootstrap import warm_technology_selection_resources
from src.infrastructure.checkpointing.mysql import open_mysql_checkpointer
from src.infrastructure.memory.milvus import MilvusMemoryIndex
from src.infrastructure.memory.mysql import MySQLMemoryRepository
from src.observability.logging import log
from src.workflows.memory_runner import MemoryAwareRunner
from src.workflows.query_router_graph import build_query_router_graph


@contextmanager
def open_memory_application(
    *,
    checkpoint_uri: str = MYSQL_CHECKPOINT_URI,
    memory_uri: str = MYSQL_MEMORY_URI,
    semantic_memory_reader: Any | None = None,
    setup_checkpoint: bool = False,
) -> Iterator[MemoryAwareRunner]:
    """Open the production graph without hidden schema creation or fallback."""
    repository = MySQLMemoryRepository(memory_uri)
    if semantic_memory_reader is None:
        semantic_memory_reader = MilvusMemoryIndex()
    if WARMUP_ON_START:
        warmup = warm_technology_selection_resources(semantic_memory_reader)
        log.info(
            "[启动][只读预热] 完成 | vector={} | graph={} | memory={}",
            warmup.get("vector", "unknown"),
            warmup.get("graph", "unknown"),
            warmup.get("memory", "unknown"),
        )
    with open_mysql_checkpointer(
        checkpoint_uri,
        setup=setup_checkpoint,
    ) as checkpointer:
        graph = build_query_router_graph(
            checkpointer,
            memory_repository=repository,
            semantic_memory_reader=semantic_memory_reader,
        )
        yield MemoryAwareRunner(
            graph=graph,
            checkpointer=checkpointer,
        )
