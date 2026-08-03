"""Zhipu web search adapter with a stable result contract."""

import json
from urllib.request import Request, urlopen

from src.config.settings import WEB_SEARCH_MAX_RESULTS, ZHIPU_API_KEY


_WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
_WEB_SEARCH_TIMEOUT_SECONDS = 30


class WebSearchError(RuntimeError):
    """Raised when Zhipu cannot return usable search results."""


def _request_web_search(
    query,
    api_key,
    max_results,
    search_engine="search_std",
):
    payload = json.dumps(
        {
            "search_query": query,
            "search_engine": search_engine,
            "search_intent": False,
            "count": max_results,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        _WEB_SEARCH_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=_WEB_SEARCH_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


class ZhipuSearch:
    """Normalize Zhipu results or raise a controlled error."""

    def __init__(self, request=None, api_key=ZHIPU_API_KEY):
        self._request = request or _request_web_search
        self._api_key = api_key

    def invoke(self, query: str, max_results=None, search_engine=None):
        if not self._api_key:
            raise WebSearchError("ZHIPU_API_KEY is not configured")

        if max_results is None:
            max_results = WEB_SEARCH_MAX_RESULTS

        try:
            if search_engine is None:
                response = self._request(
                    query,
                    self._api_key,
                    max_results,
                )
            else:
                response = self._request(
                    query,
                    self._api_key,
                    max_results,
                    search_engine,
                )
        except Exception as error:
            raise WebSearchError("Zhipu search request failed") from error

        if not isinstance(response, dict) or response.get("error"):
            raise WebSearchError("Zhipu returned an error response")
        results = response.get("search_result")
        if not isinstance(results, list) or not all(
            isinstance(result, dict) for result in results
        ):
            raise WebSearchError("Zhipu returned an unexpected response")
        return [
            {
                "title": str(result.get("title", "")).strip(),
                "content": str(result.get("content", "")).strip(),
                "url": str(result.get("link", "")).strip(),
            }
            for result in results
        ]


web_search_tool = ZhipuSearch()
