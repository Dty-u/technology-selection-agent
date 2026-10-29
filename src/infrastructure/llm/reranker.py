"""SiliconFlow rerank API client."""

import httpx

from src.config.settings import (
    RERANKER_BASE_URL,
    RERANKER_MAX_RETRIES,
    RERANKER_MODEL_NAME,
    RERANKER_TIMEOUT_SECONDS,
    SILICONFLOW_API_KEY,
)
from src.observability.logging import log

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class SiliconFlowReranker:
    """Score (query, passage) pairs with a hosted SiliconFlow reranker."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        *,
        api_key: str | None = SILICONFLOW_API_KEY,
        base_url: str = RERANKER_BASE_URL,
        timeout_seconds: float = RERANKER_TIMEOUT_SECONDS,
        max_retries: int = RERANKER_MAX_RETRIES,
        client: httpx.Client | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/rerank"
        self.models_endpoint = f"{base_url.rstrip('/')}/models"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_seconds)
        return self._client

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not configured")

    def warm(self) -> bool:
        """Open the pooled connection with an unbilled request at startup.

        The first request in a process pays DNS, CA loading, and the TLS
        handshake (~0.7s); doing it here keeps that off the first user query.
        A failed warm-up only logs, so a network blip never disables reranking.
        """
        self._require_api_key()
        try:
            self.client.get(
                self.models_endpoint,
                params={"sub_type": "reranker"},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except httpx.HTTPError as error:
            log.warning(f"---精排服务预连接失败：{type(error).__name__}---")
        return True

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    self.endpoint, json=payload, headers=headers
                )
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                continue
            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("rerank request exhausted retries")

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._require_api_key()
        body = self._post(
            {
                "model": self.model_name,
                "query": query,
                "documents": passages,
                "top_n": len(passages),
                "return_documents": False,
            }
        )
        scores = [float("-inf")] * len(passages)
        for item in body.get("results", []):
            index = int(item["index"])
            if 0 <= index < len(passages):
                scores[index] = float(item["relevance_score"])
        return scores


def create_reranker() -> SiliconFlowReranker:
    return SiliconFlowReranker()
