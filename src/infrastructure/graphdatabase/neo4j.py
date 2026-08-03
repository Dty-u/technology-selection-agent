"""Lazy Neo4j graph connection."""

from src.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


class Neo4jDatabase:
    """Create and close one lazy Neo4j graph connection."""

    def __init__(self):
        self._graph = None

    def connect(self):
        """Return a Neo4j graph connection from environment configuration."""
        if self._graph is not None:
            return self._graph

        missing_settings = [
            name
            for name, value in (
                ("NEO4J_URI", NEO4J_URI),
                ("NEO4J_USERNAME", NEO4J_USERNAME),
                ("NEO4J_PASSWORD", NEO4J_PASSWORD),
            )
            if not value
        ]
        if missing_settings:
            raise RuntimeError(
                f"Missing Neo4j environment variables: {', '.join(missing_settings)}"
            )

        try:
            from langchain_neo4j import Neo4jGraph
        except ImportError as error:
            raise RuntimeError(
                "Install neo4j and langchain-neo4j before using the graph database."
            ) from error

        self._graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            refresh_schema=False,
        )
        return self._graph

    def close(self):
        """Close the active connection, if one was created."""
        if self._graph is not None:
            self._graph.close()
            self._graph = None
