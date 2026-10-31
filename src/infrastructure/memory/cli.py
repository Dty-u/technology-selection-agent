"""Explicit setup, approval, and one-shot worker commands."""

import argparse
import json

from src.infrastructure.checkpointing.mysql import open_mysql_checkpointer
from src.infrastructure.memory.milvus import MilvusMemoryIndex
from src.infrastructure.memory.mysql import MySQLMemoryRepository
from src.infrastructure.memory.worker import MemoryProjectionWorker
from src.infrastructure.sessions.repository import MySQLSessionRepository


def setup_databases(
    *,
    checkpoint: bool,
    memory: bool,
    milvus_memory: bool = False,
    sessions: bool = False,
) -> dict[str, bool]:
    """Create only the explicitly selected schemas."""
    if not checkpoint and not memory and not milvus_memory and not sessions:
        raise ValueError(
            "select --checkpoint, --memory, --milvus-memory and/or --sessions"
        )
    result = {}
    if sessions:
        MySQLSessionRepository().setup_schema()
        result["sessions_setup"] = True
    if checkpoint:
        with open_mysql_checkpointer(setup=True):
            pass
        result["checkpoint_setup"] = True
    if memory:
        MySQLMemoryRepository().setup_schema()
        result["memory_setup"] = True
    if milvus_memory:
        result["milvus_memory_created"] = MilvusMemoryIndex().ensure_collection()
    return result


def run_worker_once(*, limit: int) -> dict[str, int]:
    repository = MySQLMemoryRepository()
    legacy_count = repository.count_projections(provider="mem0")
    if legacy_count:
        raise RuntimeError(
            "legacy Mem0 projection rows exist; inspect and migrate them explicitly "
            "before starting the direct Milvus worker"
        )
    projector = MilvusMemoryIndex()
    return MemoryProjectionWorker(repository, projector).run_once(limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="长期记忆维护命令")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="显式创建数据库表")
    setup_parser.add_argument("--checkpoint", action="store_true")
    setup_parser.add_argument("--memory", action="store_true")
    setup_parser.add_argument("--milvus-memory", action="store_true")
    setup_parser.add_argument("--sessions", action="store_true")

    worker_parser = subparsers.add_parser("worker", help="处理一批投影事件")
    worker_parser.add_argument("--limit", type=int, default=20)

    approve_parser = subparsers.add_parser("set-status", help="确认或审批记忆")
    approve_parser.add_argument("--memory-id", required=True)
    approve_parser.add_argument("--actor", required=True)
    approve_parser.add_argument(
        "--status",
        required=True,
        choices=("confirmed", "approved", "superseded", "deleted"),
    )

    args = parser.parse_args()
    if args.command == "setup":
        result = setup_databases(
            checkpoint=args.checkpoint,
            memory=args.memory,
            milvus_memory=args.milvus_memory,
            sessions=args.sessions,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command == "worker":
        print(json.dumps(run_worker_once(limit=args.limit)))
        return
    item = MySQLMemoryRepository().set_status(
        memory_id=args.memory_id,
        status=args.status,
        actor=args.actor,
    )
    print(json.dumps({"memory_id": item.memory_id, "status": item.status}))


if __name__ == "__main__":
    main()
