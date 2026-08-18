"""MySQL authoritative memory repository with transactional outbox writes."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import unquote, urlparse
from uuid import uuid4

from src.config.settings import MYSQL_MEMORY_URI
from src.config.settings import MEMORY_COLLECTION_NAME, MEMORY_OUTBOX_MAX_ATTEMPTS
from src.infrastructure.memory.models import (
    MemoryCandidate,
    MemoryItem,
    MemoryStatus,
    OutboxEvent,
)


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def parse_mysql_uri(database_uri: str) -> dict[str, Any]:
    """Parse a MySQL URI into safe PyMySQL keyword arguments."""
    parsed = urlparse(str(database_uri or "").strip())
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError("MYSQL_MEMORY_URI must use mysql:// or mysql+pymysql://")
    database = parsed.path.lstrip("/")
    if not database:
        raise ValueError("MYSQL_MEMORY_URI must include a database name")
    if not parsed.hostname or not parsed.username:
        raise ValueError("MYSQL_MEMORY_URI must include host and user")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": unquote(database),
        "charset": "utf8mb4",
        "autocommit": False,
    }


def _load_pymysql():
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError(
            "PyMySQL is required; install "
            "langgraph-checkpoint-mysql[pymysql]==3.0.0"
        ) from exc
    return pymysql


class MySQLMemoryRepository:
    """Persist authoritative memory and its projection request atomically."""

    def __init__(
        self,
        database_uri: str = MYSQL_MEMORY_URI,
        *,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        normalized_uri = str(database_uri or "").strip()
        if not normalized_uri:
            raise ValueError("MYSQL_MEMORY_URI is required for long-term memory")
        self._connection_options = parse_mysql_uri(normalized_uri)
        self._connection_factory = connection_factory

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory()
        pymysql = _load_pymysql()
        return pymysql.connect(
            **self._connection_options,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def setup_schema(self) -> None:
        """Create business-memory tables; call only from an explicit setup job."""
        statements = [
            statement.strip()
            for statement in _SCHEMA_PATH.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        with self._transaction() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def save_candidate(self, candidate: MemoryCandidate) -> MemoryItem:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM memory_items WHERE idempotency_key = %s",
                (candidate.idempotency_key,),
            )
            existing = cursor.fetchone()
            if existing:
                return self._row_to_item(existing)

            item = MemoryItem(**candidate.model_dump())
            now = item.created_at.replace(tzinfo=None)
            metadata_json = json.dumps(
                item.metadata,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cursor.execute(
                """INSERT INTO memory_items (
                    memory_id, idempotency_key, user_id, thread_id, project_id,
                    memory_type, content, status, source_kind, confidence,
                    metadata_json, version, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    item.memory_id,
                    item.idempotency_key,
                    item.user_id,
                    item.thread_id,
                    item.project_id,
                    item.memory_type,
                    item.content,
                    item.status,
                    item.source_kind,
                    item.confidence,
                    metadata_json,
                    item.version,
                    now,
                    now,
                ),
            )
            cursor.execute(
                """INSERT INTO memory_versions (
                    memory_id, version, content, status, metadata_json,
                    changed_by, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    item.memory_id,
                    item.version,
                    item.content,
                    item.status,
                    metadata_json,
                    item.source_kind,
                    now,
                ),
            )
            event = OutboxEvent(
                memory_id=item.memory_id,
                payload=item.model_dump(mode="json"),
            )
            cursor.execute(
                """INSERT INTO memory_outbox (
                    event_id, memory_id, event_type, status, attempts,
                    payload_json, available_at, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    event.event_id,
                    event.memory_id,
                    event.event_type,
                    event.status,
                    event.attempts,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.available_at.replace(tzinfo=None),
                    event.created_at.replace(tzinfo=None),
                    event.created_at.replace(tzinfo=None),
                ),
            )
            cursor.execute(
                """INSERT INTO memory_audit_events (
                    audit_id, memory_id, user_id, thread_id, action,
                    details_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(uuid4()),
                    item.memory_id,
                    item.user_id,
                    item.thread_id,
                    "memory.created",
                    json.dumps(
                        {
                            "status": item.status,
                            "memory_type": item.memory_type,
                        },
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )
            return item

    def search_authoritative(
        self,
        *,
        user_id: str,
        project_id: str | None,
        memory_types: Sequence[str],
        statuses: Sequence[MemoryStatus] = ("confirmed", "approved"),
        limit: int = 8,
    ) -> list[MemoryItem]:
        if not memory_types or not statuses or limit <= 0:
            return []
        type_marks = ", ".join(["%s"] * len(memory_types))
        status_marks = ", ".join(["%s"] * len(statuses))
        project_clause = "project_id IS NULL"
        parameters: list[Any] = [user_id]
        if project_id is not None:
            project_clause = "project_id = %s"
            parameters.append(project_id)
        parameters.extend(memory_types)
        parameters.extend(statuses)
        parameters.append(int(limit))
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT * FROM memory_items
                    WHERE user_id = %s AND {project_clause}
                      AND memory_type IN ({type_marks})
                      AND status IN ({status_marks})
                    ORDER BY updated_at DESC LIMIT %s""",
                    tuple(parameters),
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
        return [self._row_to_item(row) for row in rows]

    def get_by_ids(self, memory_ids: Sequence[str]) -> list[MemoryItem]:
        """Load authoritative items in semantic-hit order."""
        normalized_ids = [
            memory_id
            for memory_id in dict.fromkeys(
                str(memory_id or "").strip() for memory_id in memory_ids
            )
            if memory_id
        ]
        if not normalized_ids:
            return []
        marks = ", ".join(["%s"] * len(normalized_ids))
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM memory_items WHERE memory_id IN ({marks})",
                    tuple(normalized_ids),
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
        items_by_id = {
            item.memory_id: item for item in map(self._row_to_item, rows)
        }
        return [
            items_by_id[memory_id]
            for memory_id in normalized_ids
            if memory_id in items_by_id
        ]

    def claim_outbox_batch(self, *, limit: int = 20) -> list[OutboxEvent]:
        """Atomically claim ready projection events for one worker."""
        if limit <= 0:
            return []
        with self._transaction() as cursor:
            cursor.execute(
                """SELECT * FROM memory_outbox
                WHERE status = 'pending' AND available_at <= UTC_TIMESTAMP(6)
                ORDER BY created_at LIMIT %s FOR UPDATE SKIP LOCKED""",
                (int(limit),),
            )
            rows = list(cursor.fetchall())
            for row in rows:
                cursor.execute(
                    """UPDATE memory_outbox
                    SET status = 'processing', attempts = attempts + 1,
                        updated_at = UTC_TIMESTAMP(6)
                    WHERE event_id = %s""",
                    (row["event_id"],),
                )
                row["status"] = "processing"
                row["attempts"] = int(row.get("attempts", 0)) + 1
        return [self._row_to_outbox(row) for row in rows]

    def set_status(
        self,
        *,
        memory_id: str,
        status: MemoryStatus,
        actor: str,
    ) -> MemoryItem:
        """Apply an explicit confirmation/approval transition with audit."""
        if status not in {
            "pending",
            "confirmed",
            "approved",
            "superseded",
            "deleted",
        }:
            raise ValueError(f"unsupported memory status: {status}")
        normalized_actor = str(actor or "").strip()
        if not normalized_actor:
            raise ValueError("actor must not be empty")
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM memory_items WHERE memory_id = %s FOR UPDATE",
                (memory_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"memory not found: {memory_id}")
            item = self._row_to_item(row)
            next_version = item.version + 1
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            cursor.execute(
                """UPDATE memory_items
                SET status = %s, version = %s, updated_at = %s
                WHERE memory_id = %s""",
                (status, next_version, now, memory_id),
            )
            cursor.execute(
                """INSERT INTO memory_versions (
                    memory_id, version, content, status, metadata_json,
                    changed_by, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    memory_id,
                    next_version,
                    item.content,
                    status,
                    json.dumps(item.metadata, ensure_ascii=False),
                    normalized_actor,
                    now,
                ),
            )
            updated = item.model_copy(
                update={
                    "status": status,
                    "version": next_version,
                    "updated_at": now.replace(tzinfo=timezone.utc),
                }
            )
            event = OutboxEvent(
                memory_id=memory_id,
                event_type=(
                    "memory.delete" if status == "deleted" else "memory.upsert"
                ),
                payload=updated.model_dump(mode="json"),
            )
            cursor.execute(
                """INSERT INTO memory_outbox (
                    event_id, memory_id, event_type, status, attempts,
                    payload_json, available_at, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    event.event_id,
                    event.memory_id,
                    event.event_type,
                    event.status,
                    event.attempts,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.available_at.replace(tzinfo=None),
                    event.created_at.replace(tzinfo=None),
                    event.created_at.replace(tzinfo=None),
                ),
            )
            cursor.execute(
                """INSERT INTO memory_audit_events (
                    audit_id, memory_id, user_id, thread_id, action,
                    details_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(uuid4()),
                    memory_id,
                    item.user_id,
                    item.thread_id,
                    "memory.status_changed",
                    json.dumps(
                        {"from": item.status, "to": status, "actor": normalized_actor},
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )
            return updated

    def mark_outbox_projected(
        self,
        *,
        event_id: str,
        memory_id: str,
        external_id: str,
        provider: str,
    ) -> None:
        """Record the projection mapping and finish one outbox event."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT version FROM memory_items WHERE memory_id = %s",
                (memory_id,),
            )
            row = cursor.fetchone() or {"version": 1}
            cursor.execute(
                """INSERT INTO memory_projections (
                    memory_id, provider, external_id, collection_name,
                    projected_version, projected_at
                ) VALUES (%s, %s, %s, %s, %s, UTC_TIMESTAMP(6))
                ON DUPLICATE KEY UPDATE
                    external_id = VALUES(external_id),
                    collection_name = VALUES(collection_name),
                    projected_version = VALUES(projected_version),
                    projected_at = VALUES(projected_at)""",
                (
                    memory_id,
                    provider,
                    external_id,
                    MEMORY_COLLECTION_NAME,
                    int(row["version"]),
                ),
            )
            cursor.execute(
                """UPDATE memory_outbox
                SET status = 'projected', last_error = NULL,
                    updated_at = UTC_TIMESTAMP(6)
                WHERE event_id = %s""",
                (event_id,),
            )

    def mark_outbox_failed(self, *, event_id: str, error: str) -> None:
        """Schedule a bounded exponential retry or dead-letter the event."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT attempts FROM memory_outbox WHERE event_id = %s FOR UPDATE",
                (event_id,),
            )
            row = cursor.fetchone()
            if not row:
                return
            attempts = int(row["attempts"])
            if attempts >= MEMORY_OUTBOX_MAX_ATTEMPTS:
                cursor.execute(
                    """UPDATE memory_outbox
                    SET status = 'failed', last_error = %s,
                        updated_at = UTC_TIMESTAMP(6)
                    WHERE event_id = %s""",
                    (str(error)[:2000], event_id),
                )
                return
            delay_seconds = min(2 ** max(attempts, 1), 300)
            cursor.execute(
                """UPDATE memory_outbox
                SET status = 'pending', last_error = %s,
                    available_at = DATE_ADD(UTC_TIMESTAMP(6), INTERVAL %s SECOND),
                    updated_at = UTC_TIMESTAMP(6)
                WHERE event_id = %s""",
                (str(error)[:2000], delay_seconds, event_id),
            )

    def count_projections(self, *, provider: str) -> int:
        """Read-only cutover guard for projections created by another provider."""
        normalized_provider = str(provider or "").strip()
        if not normalized_provider:
            raise ValueError("provider must not be empty")
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT COUNT(*) AS projection_count
                    FROM memory_projections WHERE provider = %s""",
                    (normalized_provider,),
                )
                row = cursor.fetchone() or {"projection_count": 0}
        finally:
            connection.close()
        return int(row["projection_count"])

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> MemoryItem:
        values = dict(row)
        metadata = values.pop("metadata_json", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        values["metadata"] = metadata or {}
        for field in ("created_at", "updated_at"):
            value = values.get(field)
            if isinstance(value, datetime) and value.tzinfo is None:
                values[field] = value.replace(tzinfo=timezone.utc)
        return MemoryItem.model_validate(values)

    @staticmethod
    def _row_to_outbox(row: dict[str, Any]) -> OutboxEvent:
        values = dict(row)
        payload = values.pop("payload_json", {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        values["payload"] = payload or {}
        values.pop("last_error", None)
        values.pop("updated_at", None)
        for field in ("available_at", "created_at"):
            value = values.get(field)
            if isinstance(value, datetime) and value.tzinfo is None:
                values[field] = value.replace(tzinfo=timezone.utc)
        return OutboxEvent.model_validate(values)
