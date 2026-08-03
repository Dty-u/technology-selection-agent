"""Safe, reusable console progress logging for synchronous workflow stages."""

from contextlib import contextmanager
from functools import wraps
from threading import Event, Thread
from time import perf_counter

from src.observability.logging import log


DEFAULT_HEARTBEAT_SECONDS = 30.0


def _details_text(details):
    if not details:
        return ""
    fields = [
        f"{name}={value}"
        for name, value in details.items()
        if value is not None
    ]
    return f" | {' | '.join(fields)}" if fields else ""


def safe_progress_details(state):
    """Return non-sensitive counters and decisions suitable for console logs."""
    if not isinstance(state, dict):
        return {}
    details = {}
    scalar_fields = (
        ("status", "状态"),
        ("query_type", "类型"),
        ("source_preference", "来源"),
        ("direct_response_kind", "响应类型"),
        ("evidence_round", "证据轮次"),
        ("review_round", "审查轮次"),
    )
    for source_name, display_name in scalar_fields:
        value = state.get(source_name)
        if value is not None:
            details[display_name] = value

    requirements = state.get("requirements")
    if isinstance(requirements, dict):
        requested_candidates = requirements.get("requested_candidates")
        if isinstance(requested_candidates, list):
            details["候选请求"] = len(requested_candidates)

    list_fields = (
        ("feasible_candidates", "可行候选"),
        ("candidates", "候选"),
        ("evidence_tasks", "任务"),
        ("evidence_records", "证据"),
        ("evidence_gaps", "缺口"),
        ("evaluations", "评分"),
        ("risk_items", "风险项"),
        ("warnings", "警告"),
    )
    for source_name, display_name in list_fields:
        value = state.get(source_name)
        if isinstance(value, list):
            details[display_name] = len(value)

    review = state.get("recommendation_review")
    if isinstance(review, dict) and review.get("verdict"):
        details["审查结果"] = review["verdict"]

    report = state.get("report")
    if isinstance(report, str):
        details["报告字符数"] = len(report)
    return details


class _ProgressStage:
    def __init__(self):
        self.result_details = {}

    def set_result(self, details=None, **fields):
        if details:
            self.result_details.update(details)
        self.result_details.update(fields)


@contextmanager
def progress_stage(
    scope,
    stage_id,
    action,
    *,
    details=None,
    heartbeat_seconds=DEFAULT_HEARTBEAT_SECONDS,
):
    """Log start, heartbeat, completion, and failure for one sync operation."""
    started_at = perf_counter()
    stopped = Event()
    stage = _ProgressStage()
    log.info(
        "[{}][{}] 开始：{}{}",
        scope,
        stage_id,
        action,
        _details_text(details),
    )

    heartbeat_thread = None
    if heartbeat_seconds is not None and heartbeat_seconds > 0:
        def emit_heartbeats():
            while not stopped.wait(heartbeat_seconds):
                elapsed = perf_counter() - started_at
                log.warning(
                    "[{}][{}] 仍在执行：{} | 已等待={:.2f}s",
                    scope,
                    stage_id,
                    action,
                    elapsed,
                )

        heartbeat_thread = Thread(
            target=emit_heartbeats,
            name=f"workflow-heartbeat-{stage_id}",
            daemon=True,
        )
        heartbeat_thread.start()

    try:
        yield stage
    except BaseException as exc:
        elapsed = perf_counter() - started_at
        log.error(
            "[{}][{}] 失败：{} | 异常类型={} | 耗时={:.2f}s",
            scope,
            stage_id,
            action,
            type(exc).__name__,
            elapsed,
        )
        raise
    else:
        elapsed = perf_counter() - started_at
        log.info(
            "[{}][{}] 完成：{}{} | 耗时={:.2f}s",
            scope,
            stage_id,
            action,
            _details_text(stage.result_details),
            elapsed,
        )
    finally:
        stopped.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)


def logged_node(
    scope,
    node_name,
    action,
    node,
    *,
    heartbeat_seconds=DEFAULT_HEARTBEAT_SECONDS,
    start_details=None,
    result_details=None,
):
    """Wrap one synchronous LangGraph node without changing its state contract."""
    @wraps(node)
    def wrapped(state):
        initial_details = safe_progress_details(state)
        if start_details is not None:
            initial_details.update(start_details(state))
        with progress_stage(
            scope,
            node_name,
            action,
            details=initial_details,
            heartbeat_seconds=heartbeat_seconds,
        ) as stage:
            result = node(state)
            completion_details = safe_progress_details(result)
            if result_details is not None:
                completion_details.update(result_details(result))
            stage.set_result(completion_details)
            return result

    return wrapped
