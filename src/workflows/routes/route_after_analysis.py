def route_after_analysis(state):
    """Fan local questions out to both retrievers, or use Web search."""
    if state.get("source_preference") == "local":
        return ["vector_retrieve", "graph_retrieve"]
    route = state["route"]
    if route == "local_hybrid":
        return ["vector_retrieve", "graph_retrieve"]
    if route == "web_search":
        return "web_search"
    raise ValueError(f"Unsupported route: {route}")
