"""Entry route for the ordinary RAG question-answering workflow."""


def route_rag_qa_entry(state):
    """Bypass local analysis only when Web evidence is explicitly required."""
    if state.get("source_preference", "auto") == "web":
        return "web_search"
    return "analyze_question"

