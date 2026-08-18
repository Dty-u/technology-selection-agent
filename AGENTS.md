# Repository Guidelines

## Project Structure & Module Organization

Python code lives under `src/`. `src/workflows/` contains the LangGraph definition, shared state, execution nodes, routing decisions, and graders. External integrations belong in `src/infrastructure/`: LLM and embedding clients, Milvus access, Tavily search, and document ingestion. Configuration is loaded by `src/config/`, while logging is centralized in `src/observability/`.

Tests live in `tests/` and mirror behavior rather than directory structure. Runtime logs are written to `runtime/logs/`. The root-level `半导体技术知识库数据总结.md` is reference content, not executable source.

## Build, Test, and Development Commands

Run commands from the repository root with the project environment activated:

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_imports.py" -v
python -m unittest discover -s tests -p "test_milvus.py" -v
python -m unittest discover -s tests -p "test_retriever.py" -v
python -c "from src.workflows.query_router_graph import query_router_graph; print(type(query_router_graph).__name__)"
```

Run test files in separate Python processes; combined discovery can reload Windows native extensions and fail independently of application behavior. Start the interactive workflow with `python -m src.workflows.rag_qa_graph` only when credentials and required services are available.

## Workflow Entry Points

Use `src.workflows.application.open_memory_application` as the production
entry point. It injects the MySQL checkpointer, authoritative memory repository,
and direct Milvus semantic-memory reader. Every invocation must provide explicit
`user_id`, `thread_id`, and a new `request_id`; technology selection also requires
`project_id`:

```python
from uuid import uuid4

from src.workflows.application import open_memory_application

config = {
    "configurable": {
        "user_id": "user-001",
        "thread_id": "thread-001",
        "request_id": str(uuid4()),
        "project_id": "project-001",
    }
}
with open_memory_application() as application:
    result = application.invoke({"query": "用户的一句话"}, config=config)
```

Run the memory-aware interactive CLI with:

```powershell
python -m src.workflows.query_router_graph --user-id user-001 --project-id project-001
```

The module-level `src.workflows.query_router_graph.query_router_graph` remains a
memoryless compatibility graph for focused tests and integrations that inject no
database adapters.

The two child workflows remain independently callable for focused use cases:

```python
from src.workflows.rag_qa_graph import graph
from src.workflows.technology_selection_graph import technology_selection_graph
```

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 conventions. Name modules, functions, and variables with `snake_case`; classes use `PascalCase`; constants use `UPPER_CASE`. Use absolute project imports such as `from src.config.settings import MILVUS_URI`. Keep workflow nodes focused on state transitions and place provider-specific code under `infrastructure/`. Avoid network or database connections during module import.

## Testing Guidelines

Tests use Python's `unittest` and follow `test_<behavior>.py` naming. Mock LLMs, embeddings, Milvus, and web search by default. Add a regression test for every bug fix. Routine tests must not contact external providers, rebuild collections, or require secrets.

## Commit & Pull Request Guidelines

This checkout has no Git history from which to infer conventions. Use concise imperative commits, preferably `feat:`, `fix:`, `refactor:`, `test:`, or `docs:` prefixes. Pull requests should explain the behavior change, list verification commands, identify configuration changes, and link relevant issues.

## Security & Configuration

Keep secrets in `.env`; commit only variable names to `.env.example`. Treat `MilvusVectorSave.create_collection()` as destructive because it replaces the existing collection. Never use it as a connectivity check.
