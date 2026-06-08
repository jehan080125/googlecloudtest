#!/usr/bin/env python3
"""Generate PDF from disaster_epitaph trial summary markdown."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "disaster_epitaph_trials_summary.md"
PDF_PATH = ROOT / "disaster_epitaph_trials_summary.pdf"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunsl.ttf"),
    Path(r"C:\Windows\Fonts\NanumGothic.ttf"),
]


def find_font() -> Path:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Korean TTF font not found (malgun.ttf etc.)")


class SummaryPDF(FPDF):
    def __init__(self, font_path: Path):
        super().__init__()
        self.font_path = font_path
        self.add_font("Korean", "", str(font_path))
        self.add_font("Korean", "B", str(font_path))
        self.set_auto_page_break(auto=True, margin=18)

    def write_line(self, text: str, size: int = 10, bold: bool = False, gap: float = 5):
        self.set_font("Korean", "B" if bold else "", size)
        self.multi_cell(0, gap, text)
        self.ln(1)

    def write_table_row(self, cols: list[str], widths: list[int], header: bool = False):
        self.set_font("Korean", "B" if header else "", 9)
        h = 7
        x0 = self.get_x()
        y0 = self.get_y()
        max_h = h
        for i, (col, w) in enumerate(zip(cols, widths)):
            x = x0 + sum(widths[:i])
            self.set_xy(x, y0)
            self.multi_cell(w, h, col, border=1)
            max_h = max(max_h, self.get_y() - y0)
        self.set_xy(x0, y0 + max_h)


def parse_and_build(pdf: SummaryPDF, md_text: str):
    pdf.add_page()
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            i += 1
            continue

        if line.startswith("# "):
            pdf.write_line(line[2:], size=16, bold=True, gap=8)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.write_line(line[3:], size=13, bold=True, gap=7)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.write_line(line[4:], size=11, bold=True, gap=6)
        elif line.startswith("> "):
            pdf.set_text_color(60, 60, 60)
            pdf.write_line(line[2:], size=9)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("---"):
            pdf.ln(2)
        elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            # table
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = lines[i]
                if set(row.replace("|", "").replace("-", "").replace(":", "").strip()) <= {""}:
                    i += 1
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append(cells)
                i += 1
            if rows:
                n = len(rows[0])
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                widths = [int(page_w / n)] * n
                for ri, row in enumerate(rows):
                    while len(row) < n:
                        row.append("")
                    pdf.write_table_row(row[:n], widths, header=(ri == 0))
                pdf.ln(2)
            continue
        elif line.startswith("- ") or line.startswith("* "):
            pdf.write_line("  • " + line[2:], size=9)
        elif line.startswith("1. ") or (len(line) > 2 and line[0].isdigit() and line[1:3] == ". "):
            pdf.write_line("  " + line, size=9)
        elif line.startswith("**") and line.endswith("**"):
            pdf.write_line(line.strip("*"), size=10, bold=True)
        elif line.startswith("*") and line.endswith("*"):
            pdf.write_line(line.strip("*"), size=9)
        elif line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            for b in block:
                pdf.write_line(b, size=9)
        else:
            pdf.write_line(line, size=10)

        i += 1


def main():
    font = find_font()
    md = MD_PATH.read_text(encoding="utf-8")
    pdf = SummaryPDF(font)
    parse_and_build(pdf, md)
    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
