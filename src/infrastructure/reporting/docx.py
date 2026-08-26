"""Convert a normalized Markdown report into a professional DOCX artifact."""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_SAFE_FILENAME_PATTERN = re.compile(r"[^\w\u4e00-\u9fff-]+", re.UNICODE)


def _set_run_font(run, size=None, bold=None, color=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def _configure_styles(document):
    palette_blue = (31, 78, 121)
    body = document.styles["Normal"]
    body.font.name = "Calibri"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    body.font.size = Pt(11)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.line_spacing = 1.10

    style_specs = {
        "Title": (20, palette_blue, 0, 12),
        "Heading 1": (16, palette_blue, 16, 8),
        "Heading 2": (13, palette_blue, 12, 6),
        "Heading 3": (12, (44, 62, 80), 10, 4),
    }
    for style_name, (size, color, before, after) in style_specs.items():
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(*color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for list_style_name in ("List Bullet", "List Number"):
        list_style = document.styles[list_style_name]
        list_style.font.name = "Calibri"
        list_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        list_style.font.size = Pt(11)
        list_style.paragraph_format.left_indent = Inches(0.5)
        list_style.paragraph_format.first_line_indent = Inches(-0.25)
        list_style.paragraph_format.space_after = Pt(6)


def _configure_page(document):
    section = document.sections[0]
    section.start_type = WD_SECTION.NEW_PAGE
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header.paragraphs[0]
    header.text = "技术选型报告"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if header.runs:
        _set_run_font(header.runs[0], size=8, color=(112, 112, 112))

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("内部决策支持")
    _set_run_font(run, size=8, color=(128, 128, 128))


def _plain_text(markdown_text):
    text = _LINK_PATTERN.sub(r"\1 (\2)", markdown_text)
    return (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
        .replace(r"\|", "|")
        .strip()
    )


def _add_paragraph(document, text, style=None):
    paragraph = document.add_paragraph(style=style)
    run = paragraph.add_run(_plain_text(text))
    _set_run_font(run)
    return paragraph


def _split_table_row(line):
    """Split one Markdown table row while preserving escaped pipe characters."""
    content = line.strip()
    if content.startswith("|"):
        content = content[1:]
    if content.endswith("|") and not content.endswith(r"\|"):
        content = content[:-1]

    cells = []
    current = []
    escaped = False
    for character in content:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _is_table_separator(line):
    cells = _split_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
        for cell in cells
    )


def _append_table(document, header_cells, rows):
    column_count = len(header_cells)
    table = document.add_table(rows=1, cols=column_count)
    table.style = "Table Grid"
    for index, value in enumerate(header_cells):
        paragraph = table.rows[0].cells[index].paragraphs[0]
        run = paragraph.add_run(_plain_text(value))
        _set_run_font(run, bold=True)

    for row in rows:
        table_row = table.add_row()
        for index in range(column_count):
            value = row[index] if index < len(row) else ""
            paragraph = table_row.cells[index].paragraphs[0]
            run = paragraph.add_run(_plain_text(value))
            _set_run_font(run)


def _append_markdown(document, markdown):
    title_written = False
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if not line:
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1].strip())
        ):
            header_cells = _split_table_row(line)
            rows = []
            index += 2
            while index < len(lines):
                row_line = lines[index].strip()
                if not row_line or "|" not in row_line:
                    break
                rows.append(_split_table_row(row_line))
                index += 1
            _append_table(document, header_cells, rows)
            continue
        if line.startswith("### "):
            _add_paragraph(document, line[4:], "Heading 3")
        elif line.startswith("## "):
            _add_paragraph(document, line[3:], "Heading 2")
        elif line.startswith("# "):
            style = "Title" if not title_written else "Heading 1"
            _add_paragraph(document, line[2:], style)
            title_written = True
        elif line.startswith(('- ', '* ')):
            _add_paragraph(document, line[2:], "List Bullet")
        elif re.match(r"^\d+\.\s+", line):
            _add_paragraph(document, re.sub(r"^\d+\.\s+", "", line), "List Number")
        elif line.startswith("> "):
            paragraph = _add_paragraph(document, line[2:])
            paragraph.paragraph_format.left_indent = Inches(0.35)
            paragraph.paragraph_format.right_indent = Inches(0.2)
        else:
            _add_paragraph(document, line)
        index += 1


def _safe_query_slug(query):
    slug = _SAFE_FILENAME_PATTERN.sub("-", query.strip()).strip("-_")
    return slug[:32] or "report"


def export_report_docx(markdown, query, output_directory=None):
    """Write one unique DOCX file and return its absolute path."""
    if output_directory is None:
        from src.config.settings import REPORT_OUTPUT_DIR

        output_directory = REPORT_OUTPUT_DIR
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _safe_query_slug(query)

    while True:
        suffix = uuid.uuid4().hex[:8]
        target = directory / f"{slug}-{timestamp}-{suffix}.docx"
        if not target.exists():
            break
    temporary_path = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"

    document = Document()
    document.core_properties.title = _plain_text(
        next((line[2:] for line in markdown.splitlines() if line.startswith("# ")), query)
    )
    document.core_properties.subject = "技术选型决策支持报告"
    _configure_page(document)
    _configure_styles(document)
    _append_markdown(document, markdown)

    try:
        document.save(temporary_path)
        if target.exists():
            raise FileExistsError(f"Report already exists: {target}")
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(target)
