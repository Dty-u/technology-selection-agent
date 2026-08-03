"""Import Markdown documents into Neo4j as LLM-extracted graph data."""

import asyncio
import multiprocessing
from hashlib import sha256
from pathlib import Path
from queue import Empty, Full
from typing import List, Tuple, Union

from langchain_core.documents import Document

from src.infrastructure.graphdatabase.neo4j import Neo4jDatabase
from src.observability.logging import log


QUEUE_MAXSIZE = 3
PROCESS_JOIN_TIMEOUT_SECONDS = 60


def prepare_documents_for_graph_ingestion(
    documents: List[Document], source_path: Union[str, Path]
) -> List[Document]:
    """Attach stable source metadata so repeated imports merge documents."""
    resolved_source_path = Path(source_path).resolve()
    for index, document in enumerate(documents):
        document.metadata["id"] = sha256(
            f"{resolved_source_path}:{index}:{document.page_content}".encode("utf-8")
        ).hexdigest()
        document.metadata["source"] = str(resolved_source_path)
    return documents


def _entity_id(node) -> str:
    return f"{node.type}:{node.id}"


def write_graph_documents(graph, graph_documents) -> None:
    """Write graph documents with standard Cypher, without requiring APOC."""
    for graph_document in graph_documents:
        nodes = [
            {
                "id": _entity_id(node),
                "name": str(node.id),
                "type": node.type,
                "properties": node.properties,
            }
            for node in graph_document.nodes
        ]
        if nodes:
            graph.query(
                """
                UNWIND $nodes AS node
                MERGE (entity:Entity {id: node.id})
                SET entity.name = node.name,
                    entity.type = node.type
                SET entity += node.properties
                """,
                {"nodes": nodes},
            )

        if graph_document.source is not None:
            source = graph_document.source
            graph.query(
                """
                MERGE (document:Document {id: $document_id})
                SET document.text = $text,
                    document.source = $source
                WITH document
                UNWIND $entity_ids AS entity_id
                MATCH (entity:Entity {id: entity_id})
                MERGE (document)-[:MENTIONS]->(entity)
                """,
                {
                    "document_id": source.metadata["id"],
                    "text": source.page_content,
                    "source": source.metadata.get("source"),
                    "entity_ids": [node["id"] for node in nodes],
                },
            )

        relationships = [
            {
                "source_id": _entity_id(relationship.source),
                "target_id": _entity_id(relationship.target),
                "type": relationship.type,
                "properties": relationship.properties,
            }
            for relationship in graph_document.relationships
        ]
        if relationships:
            graph.query(
                """
                UNWIND $relationships AS relationship
                MATCH (source:Entity {id: relationship.source_id})
                MATCH (target:Entity {id: relationship.target_id})
                MERGE (source)-[edge:RELATED_TO {type: relationship.type}]->(target)
                SET edge += relationship.properties
                """,
                {"relationships": relationships},
            )


def _parse_markdown_documents(source_path: Path) -> List[Document]:
    """Parse one Markdown file and attach stable graph source metadata."""
    from src.infrastructure.document_processing.markdown_parser import MarkdownParser

    documents = MarkdownParser().parse_markdown_to_documents(str(source_path))
    if not documents:
        raise ValueError(f"Markdown file produced no documents: {source_path}")
    return prepare_documents_for_graph_ingestion(documents, source_path)


def _create_graph_transformer():
    """Create the graph transformer only when an import actually runs."""
    try:
        from langchain_neo4j import LLMGraphTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install neo4j and langchain-neo4j before importing graph documents."
        ) from error

    from src.infrastructure.llm.chat_model import llm

    return LLMGraphTransformer(
        llm=llm,
        ignore_tool_usage=True,
    )


def _write_graph_documents_to_neo4j(graph_documents) -> None:
    """Open one Neo4j connection, write graph documents, then close it."""
    database = Neo4jDatabase()
    graph = database.connect()
    try:
        write_graph_documents(graph, graph_documents)
    finally:
        database.close()


def _build_import_result(documents: List[Document], graph_documents) -> dict:
    entity_count = sum(len(graph_document.nodes) for graph_document in graph_documents)
    if not entity_count:
        source_path = documents[0].metadata.get("source", "unknown source")
        raise RuntimeError(f"No entities were extracted from: {source_path}")

    return {
        "chunk_count": len(documents),
        "entity_count": entity_count,
        "document_ids": [document.metadata["id"] for document in documents],
    }


def inject_markdown_file(md_file: Union[str, Path]) -> dict:
    """Parse one Markdown file, extract graph entities, and import them to Neo4j."""
    source_path = Path(md_file).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Markdown file does not exist: {source_path}")

    documents = _parse_markdown_documents(source_path)
    graph_documents = _create_graph_transformer().convert_to_graph_documents(documents)
    result = _build_import_result(documents, graph_documents)
    _write_graph_documents_to_neo4j(graph_documents)
    log.info(f"Neo4j graph import completed: {result}")
    return result


def _parse_markdown_directory_process(parsed_queue, result_queue, md_files: List[str]) -> None:
    """Parse Markdown in a dedicated process and send documents to the coordinator."""
    try:
        for file_name in md_files:
            source_path = Path(file_name)
            try:
                documents = _parse_markdown_documents(source_path)
                parsed_queue.put((str(source_path), documents))
            except Exception as error:
                result_queue.put(("failure", str(source_path), "parse", str(error)))
                log.error(f"Markdown parse failed: {source_path}: {error}")
    finally:
        parsed_queue.put(None)


def _neo4j_writer_process(graph_queue, result_queue) -> None:
    """Keep one Neo4j connection open while writing graph documents from a queue."""
    database = None
    graph = None
    connection_error = None
    try:
        database = Neo4jDatabase()
        graph = database.connect()
    except Exception as error:
        connection_error = str(error)
        log.error(f"Neo4j writer connection failed: {error}")

    try:
        while True:
            item = graph_queue.get()
            if item is None:
                break

            source_path, graph_documents, result = item
            if connection_error is not None:
                result_queue.put(("failure", source_path, "write", connection_error))
                continue

            try:
                write_graph_documents(graph, graph_documents)
                result_queue.put(("success", source_path, result))
            except Exception as error:
                result_queue.put(("failure", source_path, "write", str(error)))
                log.error(f"Neo4j graph write failed: {source_path}: {error}")
    finally:
        if database is not None:
            database.close()


def _drain_result_queue(result_queue, successes: dict, failures: dict) -> None:
    """Collect parser and writer outcomes without blocking the coordinator."""
    while True:
        try:
            item = result_queue.get_nowait()
        except Empty:
            return

        status, source_path, *payload = item
        if status == "success":
            successes[source_path] = payload[0]
            continue

        stage, message = payload
        failures[source_path] = f"{stage}: {message}"


async def _get_queue_item(queue, producer_process) -> object:
    """Wait for a queue item while detecting an unexpectedly stopped producer."""
    while True:
        try:
            return await asyncio.to_thread(queue.get, True, 0.2)
        except Empty:
            if not producer_process.is_alive():
                raise RuntimeError(
                    f"{producer_process.name} stopped before sending its completion signal."
                )


async def _put_queue_item(queue, item, consumer_process) -> None:
    """Put an item with backpressure while detecting a stopped consumer."""
    while True:
        try:
            await asyncio.to_thread(queue.put, item, True, 0.2)
            return
        except Full:
            if not consumer_process.is_alive():
                raise RuntimeError(
                    f"{consumer_process.name} stopped before consuming queued data."
                )


async def _coordinate_graph_ingestion(
    parsed_queue,
    graph_queue,
    result_queue,
    parser_process,
    writer_process,
    max_concurrency: int,
) -> Tuple[dict, dict]:
    """Extract graph documents asynchronously between parser and writer processes."""
    file_semaphore = asyncio.Semaphore(max_concurrency)
    llm_semaphore = asyncio.Semaphore(max_concurrency)
    successes = {}
    failures = {}
    extraction_tasks = set()

    async def extract_one_file(source_path: Path, documents: List[Document]) -> None:
        async with file_semaphore:
            try:
                transformer = _create_graph_transformer()

                async def extract_graph_document(document: Document):
                    async with llm_semaphore:
                        return await transformer.aprocess_response(document)

                graph_documents = await asyncio.gather(
                    *(extract_graph_document(document) for document in documents)
                )
                result = _build_import_result(documents, graph_documents)
                await _put_queue_item(
                    graph_queue,
                    (str(source_path), graph_documents, result),
                    writer_process,
                )
            except Exception as error:
                failures[str(source_path)] = f"extract: {error}"
                log.error(f"Neo4j graph extraction failed: {source_path}: {error}")

    try:
        while True:
            _drain_result_queue(result_queue, successes, failures)
            item = await _get_queue_item(parsed_queue, parser_process)
            if item is None:
                break

            source_path, documents = item
            task = asyncio.create_task(
                extract_one_file(Path(source_path), documents)
            )
            extraction_tasks.add(task)
            task.add_done_callback(extraction_tasks.discard)

        if extraction_tasks:
            await asyncio.gather(*extraction_tasks)
    finally:
        await _put_queue_item(graph_queue, None, writer_process)

    await asyncio.to_thread(parser_process.join, PROCESS_JOIN_TIMEOUT_SECONDS)
    if parser_process.is_alive():
        parser_process.terminate()
        await asyncio.to_thread(parser_process.join)
        raise RuntimeError("Markdown parser process timed out and was terminated.")
    _drain_result_queue(result_queue, successes, failures)

    await asyncio.to_thread(writer_process.join, PROCESS_JOIN_TIMEOUT_SECONDS)
    if writer_process.is_alive():
        writer_process.terminate()
        await asyncio.to_thread(writer_process.join)
        raise RuntimeError("Neo4j writer process timed out and was terminated.")

    _drain_result_queue(result_queue, successes, failures)
    return successes, failures


def _run_graph_ingestion_pipeline(md_files: List[Path], max_concurrency: int) -> Tuple[dict, dict]:
    """Start the parser and writer processes around the async LLM coordinator."""
    context = multiprocessing.get_context("spawn")
    parsed_queue = context.Queue(maxsize=QUEUE_MAXSIZE)
    graph_queue = context.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = context.Queue()
    file_names = [str(path) for path in md_files]
    parser_process = context.Process(
        name="graph-markdown-parser",
        target=_parse_markdown_directory_process,
        args=(parsed_queue, result_queue, file_names),
    )
    writer_process = context.Process(
        name="neo4j-graph-writer",
        target=_neo4j_writer_process,
        args=(graph_queue, result_queue),
    )
    parser_process.start()
    writer_process.start()

    try:
        successes, failures = asyncio.run(
            _coordinate_graph_ingestion(
                parsed_queue,
                graph_queue,
                result_queue,
                parser_process,
                writer_process,
                max_concurrency,
            )
        )
    finally:
        parser_process.join(PROCESS_JOIN_TIMEOUT_SECONDS)
        if parser_process.is_alive():
            parser_process.terminate()
            parser_process.join()
        if writer_process.is_alive():
            writer_process.terminate()
            writer_process.join()
        parsed_queue.close()
        graph_queue.close()
        result_queue.close()

    return successes, failures

def inject_markdown_directory(
    directory: Union[str, Path],
    max_concurrency: int = 3,
) -> dict:
    """Import Markdown files in one directory with bounded LLM concurrency."""
    source_directory = Path(directory).resolve()
    if not source_directory.is_dir():
        raise NotADirectoryError(f"Markdown directory does not exist: {source_directory}")
    if not isinstance(max_concurrency, int) or max_concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer")

    md_files = sorted(
        path for path in source_directory.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    if not md_files:
        raise ValueError(f"No Markdown files found in directory: {source_directory}")

    successes, failures = _run_graph_ingestion_pipeline(md_files, max_concurrency)
    result = {
        "directory": str(source_directory),
        "total_files": len(md_files),
        "success_count": len(successes),
        "failure_count": len(failures),
        "total_chunk_count": sum(item["chunk_count"] for item in successes.values()),
        "total_entity_count": sum(item["entity_count"] for item in successes.values()),
        "files": successes,
        "failures": failures,
    }
    log.info(f"Neo4j graph directory import completed: {result}")
    return result


if __name__ == "__main__":
    inject_markdown_directory(r"D:\Mydocs\rag_md\left_md")
