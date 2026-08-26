"""Adapters that keep synchronous integrations off LangGraph's event loop."""

import asyncio
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any


def async_node(node: Callable[..., Any]) -> Callable[..., Any]:
    """Expose a sync workflow node as an async LangGraph node.

    Providers in this project still expose synchronous clients. Running them
    in the default worker pool preserves their behaviour without blocking the
    ASGI event loop that serves the frontend request.
    """
    accepts_config = "config" in inspect.signature(node).parameters

    @wraps(node)
    async def wrapped(state: Any, config: Any = None) -> Any:
        if inspect.iscoroutinefunction(node):
            if accepts_config:
                return await node(state, config=config)
            return await node(state)
        if accepts_config:
            return await asyncio.to_thread(node, state, config=config)
        return await asyncio.to_thread(node, state)

    return wrapped
