import subprocess
import sys
import unittest
import ast
from pathlib import Path


class ImportLayoutTest(unittest.TestCase):
    def test_src_enables_hugging_face_offline_mode_before_hub_import(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; "
                "os.environ.pop('HF_HUB_OFFLINE', None); "
                "import src; "
                "from huggingface_hub.constants import HF_HUB_OFFLINE; "
                "print(os.environ.get('HF_HUB_OFFLINE')); "
                "print(HF_HUB_OFFLINE)",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines(), ["1", "True"])

    def test_src_packages_import_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.workflows.nodes.retrieve import retrieve; "
                "print(retrieve.__name__)",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "retrieve")

    def test_markdown_parser_does_not_shadow_third_party_markdown(self):
        project_root = Path(__file__).resolve().parents[1]
        parser_directory = (
            project_root / "src" / "infrastructure" / "document_processing"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util, sys; "
                f"sys.path.insert(0, {str(parser_directory)!r}); "
                "parser = importlib.util.find_spec("
                "'src.infrastructure.document_processing.markdown_parser'); "
                "markdown = importlib.util.find_spec('markdown'); "
                "print(parser is not None); "
                "print(markdown.submodule_search_locations is not None)",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines(),
            ["True", "True"],
        )

    def test_markdown_parser_log_messages_are_valid_chinese(self):
        parser_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "infrastructure"
            / "document_processing"
            / "markdown_parser.py"
        )
        tree = ast.parse(parser_path.read_text(encoding="utf-8"))
        log_messages = [
            value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "info"
            and node.args
            and isinstance(node.args[0], ast.JoinedStr)
            for value in node.args[0].values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]

        self.assertIn("文件解析后的docs长度: ", log_messages)
        self.assertIn("文件合并后的长度: ", log_messages)
        self.assertIn("语义切割后的长度: ", log_messages)

    def test_graph_ingestion_disables_unsupported_structured_output(self):
        ingestion_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "infrastructure"
            / "document_processing"
            / "graph_ingestion.py"
        )
        tree = ast.parse(ingestion_path.read_text(encoding="utf-8"))
        transformer_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LLMGraphTransformer"
        ]

        self.assertEqual(len(transformer_calls), 1)
        self.assertTrue(
            any(
                keyword.arg == "ignore_tool_usage"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in transformer_calls[0].keywords
            )
        )

    def test_graph_ingestion_does_not_require_apoc_schema_refresh(self):
        ingestion_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "infrastructure"
            / "document_processing"
            / "graph_ingestion.py"
        )
        tree = ast.parse(ingestion_path.read_text(encoding="utf-8"))
        add_documents_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_graph_documents"
        ]

        self.assertEqual(add_documents_calls, [])


if __name__ == "__main__":
    unittest.main()
