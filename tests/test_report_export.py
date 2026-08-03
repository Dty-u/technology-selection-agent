import unittest
from pathlib import Path

from docx import Document


class DocxReportExportTest(unittest.TestCase):
    def test_exported_docx_preserves_report_structure_and_visible_sources(self):
        from src.infrastructure.reporting.docx import export_report_docx

        markdown = """# 半导体技术选型报告

## 候选评分

### GAA

- power: 5/5，功耗和能效更适合目标负载 [EPOWER01]
- 来源：https://example.com/power
"""
        output_directory = Path.cwd() / "runtime" / "reports"
        first_path = None
        second_path = None
        try:
            first_path = Path(
                export_report_docx(
                    markdown,
                    "高端手机低功耗 SoC：FinFET 与 GAA",
                    output_directory,
                )
            )
            second_path = Path(
                export_report_docx(
                    markdown,
                    "高端手机低功耗 SoC：FinFET 与 GAA",
                    output_directory,
                )
            )

            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.parent, output_directory)
            document = Document(first_path)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            styles = {
                paragraph.text: paragraph.style.name
                for paragraph in document.paragraphs
                if paragraph.text
            }
        finally:
            for generated_path in (first_path, second_path):
                if generated_path is not None and generated_path.exists():
                    generated_path.unlink()

        self.assertIn("半导体技术选型报告", text)
        self.assertIn("power: 5/5", text)
        self.assertIn("EPOWER01", text)
        self.assertIn("https://example.com/power", text)
        self.assertEqual(styles["候选评分"], "Heading 2")
        self.assertEqual(styles["GAA"], "Heading 3")


if __name__ == "__main__":
    unittest.main()
