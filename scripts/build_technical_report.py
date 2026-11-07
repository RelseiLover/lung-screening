"""Render the adapted Markdown report to a standalone Chinese PDF.

Optional document-only dependencies are listed in requirements-docs.txt.
Windows uses installed YaHei fonts; other systems may pass font paths.
"""

import argparse
import re
from html import escape
from pathlib import Path

from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
WIDTH = A4[0] - 104


def register_fonts(regular, bold):
    if regular and Path(regular).is_file():
        bold = bold if bold and Path(bold).is_file() else regular
        pdfmetrics.registerFont(TTFont("ReportBody", str(regular), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("ReportBold", str(bold or regular), subfontIndex=0))
        return "ReportBody", "ReportBold"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light", "STSong-Light"


class NumberedCanvas(canvas.Canvas):
    footer_font = "STSong-Light"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def showPage(self):
        self.states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setFont(self.footer_font, 8)
            self.setFillColor(colors.HexColor("#66756f"))
            self.drawString(52, 27, "肺智瞳 参赛技术文档 2026年11月7日")
            self.drawRightString(A4[0] - 52, 27, f"{self._pageNumber} / {total}")
            super().showPage()
        super().save()


def architecture(font):
    drawing = Drawing(WIDTH, 214)
    ink = colors.HexColor("#21463b")
    stroke = colors.HexColor("#8da99d")

    def box(x, y, w, h, label):
        drawing.add(Rect(x, y, w, h, strokeColor=stroke, fillColor=colors.white))
        drawing.add(
            String(
                x + w / 2,
                y + h / 2 - 4,
                label,
                fontName=font,
                fontSize=10,
                textAnchor="middle",
                fillColor=ink,
            )
        )

    def arrow(x, top, bottom):
        drawing.add(Line(x, top, x, bottom, strokeColor=ink))
        drawing.add(
            Polygon(
                [x, bottom, x - 3, bottom + 5, x + 3, bottom + 5], fillColor=ink, strokeColor=ink
            )
        )

    box(12, 174, WIDTH - 24, 32, "React与TypeScript 问诊 上传 证据展示")
    arrow(WIDTH / 2, 174, 158)
    box(12, 126, WIDTH - 24, 32, "FastAPI 校验 同源接口 临时文件管理")
    arrow(WIDTH / 2, 126, 110)
    box(12, 78, WIDTH - 24, 32, "LangGraph 状态 路由 复核与反馈")
    arrow(WIDTH / 2, 78, 62)
    labels = ["本地CNN", "TF-IDF检索", "专病模型或规则", "可选DeepSeek"]
    gap = 10
    width = (WIDTH - 24 - 3 * gap) / 4
    for index, label in enumerate(labels):
        box(12 + index * (width + gap), 22, width, 40, label)
    return drawing


def render(source, output, regular=None, bold=None):
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    body_font, heading_font = register_fonts(regular, bold)
    NumberedCanvas.footer_font = body_font
    styles = {
        "body": ParagraphStyle(
            "Body",
            fontName=body_font,
            fontSize=10.7,
            leading=18.5,
            spaceAfter=11,
            wordWrap="CJK",
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
        "title": ParagraphStyle(
            "Title",
            fontName=heading_font,
            fontSize=23,
            leading=33,
            spaceAfter=20,
            wordWrap="CJK",
            textColor=colors.black,
        ),
        "chapter": ParagraphStyle(
            "Chapter",
            fontName=heading_font,
            fontSize=19,
            leading=28,
            spaceAfter=16,
            wordWrap="CJK",
            textColor=colors.black,
            keepWithNext=True,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName=heading_font,
            fontSize=12.2,
            leading=20,
            spaceBefore=9,
            spaceAfter=7,
            wordWrap="CJK",
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            fontName=heading_font,
            fontSize=11.2,
            leading=19,
            spaceBefore=7,
            spaceAfter=6,
            wordWrap="CJK",
            keepWithNext=True,
        ),
        "table": ParagraphStyle(
            "Table",
            fontName=body_font,
            fontSize=9.4,
            leading=15.5,
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName=heading_font, fontSize=9.4, leading=15.5, wordWrap="CJK"
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=body_font,
            fontSize=9,
            leading=15,
            textColor=colors.HexColor("#5e7068"),
            spaceAfter=12,
            alignment=1,
        ),
        "code": ParagraphStyle(
            "Code", fontName=body_font, fontSize=9.5, leading=16, spaceAfter=2, wordWrap="CJK"
        ),
    }
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    story = []
    first_heading = True
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- page -->":
            story.append(PageBreak())
        elif line == "[[architecture]]":
            story.append(architecture(body_font))
        elif line == "[[screenshot]]":
            asset = ROOT / "docs/assets/react-interface.png"
            if not asset.is_file():
                raise FileNotFoundError(asset)
            image = Image(str(asset))
            scale = min(WIDTH / image.imageWidth, 340 / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            story.append(image)
            story.append(Spacer(1, 6))
        elif line.startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                story.append(Paragraph(escape(lines[index]), styles["code"]))
                index += 1
            story.append(Spacer(1, 10))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [value.strip() for value in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", value) for value in cells):
                    rows.append(cells)
                index += 1
            count = len(rows[0])
            data = [
                [
                    Paragraph(
                        escape(value), styles["table_header"] if row_index == 0 else styles["table"]
                    )
                    for value in row
                ]
                for row_index, row in enumerate(rows)
            ]
            table = Table(data, colWidths=[WIDTH / count] * count, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#6e8a7e")),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9caf9f")),
                        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d5dfd9")),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
            continue
        elif line.startswith("# "):
            style = "title" if first_heading else "chapter"
            story.append(Paragraph(escape(line[2:]), styles[style]))
            first_heading = False
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["section"]))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), styles["subsection"]))
        else:
            style = "caption" if re.match(r"图\d", line) else "body"
            story.append(Paragraph(escape(line), styles[style]))
        index += 1
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=52,
        rightMargin=52,
        topMargin=48,
        bottomMargin=48,
        title="肺智瞳多模态肺部异常智能辅助筛查系统 参赛技术文档",
        author="",
        pageCompression=1,
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    reader = PdfReader(str(output))
    assert len(reader.pages) <= 30, f"Page limit exceeded: {len(reader.pages)}"
    assert all((page.extract_text() or "").strip() for page in reader.pages)
    print(f"Technical report generated: {len(reader.pages)} pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "docs/technical-report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/technical-report.pdf")
    parser.add_argument("--font-path", default="C:/Windows/Fonts/msyh.ttc")
    parser.add_argument("--bold-font-path", default="C:/Windows/Fonts/msyhbd.ttc")
    args = parser.parse_args()
    render(args.source, args.output, args.font_path, args.bold_font_path)
