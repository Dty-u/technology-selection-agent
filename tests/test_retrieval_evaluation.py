"""Deterministic hybrid-retrieval regression metrics with an opt-in live run."""

import os
import unittest

from langchain_core.documents import Document


EVALUATION_CASES = [
    {
        "id": "euv-secondary-electron-acid-diffusion",
        "question": "EUV 的高光子能量如何通过二次电子和酸扩散影响图形精度？",
        "entities": ["二次电子", "图形精度"],
        "relation_types": ["AFFECTS"],
        "expected_source_suffix": "tech_report_0ui655n3.md",
    }
]


def evaluate_case(case, vector_documents, graph_documents):
    """Score only supplied documents; this helper never creates a retriever."""
    vector_source_hit = any(
        str(document.metadata.get("source", "")).endswith(
            case["expected_source_suffix"]
        )
        for document in vector_documents[:4]
    )
    graph_path_hit = any(
        set(case["entities"]).issubset(set(document.metadata.get("entities", [])))
        and set(case["relation_types"]).issubset(
            set(document.metadata.get("relations", []))
        )
        for document in graph_documents[:4]
    )
    return {
        "id": case["id"],
        "vector_source_hit_at_4": vector_source_hit,
        "graph_path_hit_at_4": graph_path_hit,
    }


class RetrievalEvaluationTest(unittest.TestCase):
    def test_offline_metrics_report_vector_and_graph_hits(self):
        case = EVALUATION_CASES[0]
        vector_documents = [
            Document(
                page_content="vector evidence",
                metadata={"source": "data/tech_report_0ui655n3.md"},
            )
        ]
        graph_documents = [
            Document(
                page_content="graph evidence",
                metadata={
                    "entities": ["二次电子", "图形精度"],
                    "relations": ["AFFECTS"],
                }
            )
        ]

        result = evaluate_case(case, vector_documents, graph_documents)

        self.assertTrue(result["vector_source_hit_at_4"])
        self.assertTrue(result["graph_path_hit_at_4"])

    def test_empty_graph_result_is_a_valid_miss(self):
        result = evaluate_case(EVALUATION_CASES[0], [], [])

        self.assertFalse(result["vector_source_hit_at_4"])
        self.assertFalse(result["graph_path_hit_at_4"])

    @unittest.skipUnless(
        os.getenv("RUN_RETRIEVAL_EVALUATION") == "1",
        "set RUN_RETRIEVAL_EVALUATION=1 to contact Milvus and Neo4j",
    )
    def test_live_retrieval_metrics(self):
        from src.config.bootstrap import get_graph_retriever, get_retriever

        vector_retriever = get_retriever()
        graph_retriever = get_graph_retriever()
        results = []
        for case in EVALUATION_CASES:
            vector_documents = vector_retriever.invoke(case["question"])
            graph_documents = graph_retriever.invoke(
                {
                    "entities": case["entities"],
                    "relation_types": case["relation_types"],
                    "relation_intent": case["question"],
                }
            )
            results.append(evaluate_case(case, vector_documents, graph_documents))

        print({
            "case_count": len(results),
            "vector_source_hits": sum(
                result["vector_source_hit_at_4"] for result in results
            ),
            "graph_path_hits": sum(
                result["graph_path_hit_at_4"] for result in results
            ),
            "failures": [
                result["id"] for result in results
                if not (
                    result["vector_source_hit_at_4"]
                    and result["graph_path_hit_at_4"]
                )
            ],
        })


if __name__ == "__main__":
    unittest.main()
