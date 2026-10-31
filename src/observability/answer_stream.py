"""Request-scoped channel that forwards answer tokens from workflow nodes."""

from contextlib import contextmanager
from contextvars import ContextVar

from src.observability.logging import log


_answer_observer = ContextVar("answer_stream_observer", default=None)


@contextmanager
def observe_answer_stream(observer):
    """Send answer reset/delta events to one request-scoped observer."""
    token = _answer_observer.set(observer)
    try:
        yield
    finally:
        _answer_observer.reset(token)


def has_answer_observer():
    return _answer_observer.get() is not None


def _notify(event):
    observer = _answer_observer.get()
    if observer is None:
        return
    try:
        observer(event)
    except Exception as exc:
        log.warning(
            "[回答流] 观察器失败 | 事件={} | 异常类型={}",
            event.get("type"),
            type(exc).__name__,
        )


def emit_answer_reset():
    """Tell the client to discard any partial answer before a new generation."""
    _notify({"type": "reset"})


def emit_answer_delta(text):
    if text:
        _notify({"type": "delta", "text": str(text)})


def stream_chain_text(chain, inputs):
    """Run a string-output chain, streaming chunks when a client is listening."""
    if not has_answer_observer():
        return chain.invoke(inputs)
    emit_answer_reset()
    parts = []
    for chunk in chain.stream(inputs):
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
        if not isinstance(text, str) or not text:
            continue
        parts.append(text)
        emit_answer_delta(text)
    return "".join(parts)
