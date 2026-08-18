from functools import lru_cache
from time import monotonic

from src.config.settings import RESOURCE_FAILURE_COOLDOWN_SECONDS


class ResourceFailureCooldown:
    """Prevent repeated connection timeouts while allowing later recovery."""

    def __init__(self, cooldown_seconds, *, clock=monotonic):
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._failed_until = {}

    def allow(self, resource):
        return self._clock() >= self._failed_until.get(resource, 0.0)

    def mark_failed(self, resource):
        self._failed_until[resource] = self._clock() + self.cooldown_seconds

    def mark_ready(self, resource):
        self._failed_until.pop(resource, None)


resource_failure_cooldown = ResourceFailureCooldown(
    RESOURCE_FAILURE_COOLDOWN_SECONDS
)


def resource_retry_allowed(resource):
    return resource_failure_cooldown.allow(resource)


def mark_resource_failed(resource):
    resource_failure_cooldown.mark_failed(resource)


def mark_resource_ready(resource):
    resource_failure_cooldown.mark_ready(resource)


@lru_cache(maxsize=1)
def get_retriever():
    """Create the Milvus retriever on first use and reuse it afterwards."""
    from src.infrastructure.vectorstores.milvus import create_retriever

    return create_retriever()


@lru_cache(maxsize=1)
def get_graph_retriever():
    """Create the Neo4j graph retriever on first use and reuse it afterwards."""
    from src.infrastructure.graphdatabase.retriever import create_graph_retriever

    return create_graph_retriever()


def warm_technology_selection_resources(semantic_memory_reader=None):
    """Warm reusable retrieval resources during host startup without mutation."""
    result = {}
    try:
        get_retriever()
        result["vector"] = "ready"
        mark_resource_ready("vector")
    except Exception as error:
        result["vector"] = type(error).__name__
        mark_resource_failed("vector")
    try:
        get_graph_retriever()._get_graph()
        result["graph"] = "ready"
        mark_resource_ready("graph")
    except Exception as error:
        result["graph"] = type(error).__name__
        mark_resource_failed("graph")
    if semantic_memory_reader is not None:
        try:
            result["memory"] = (
                "ready" if semantic_memory_reader.warm() else "missing"
            )
            mark_resource_ready("memory")
        except Exception as error:
            result["memory"] = type(error).__name__
            mark_resource_failed("memory")
    return result
