"""Build phases/01-instrument/factworld.pdf from phases/01-instrument/factworld.md —
markdown -> styled HTML -> weasyprint PDF.

Pure-Python (no LaTeX). Single-column academic/preprint styling: Times-like serif
(Nimbus Roman) with Noto/DejaVu fallback for math glyphs, booktabs tables, first-line
indent, centered title block.

  uv pip install weasyprint markdown pygments   # one-time
  .venv/bin/python scripts/build_pdf.py
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "phases", "01-instrument", "factworld.md")
OUT = os.path.join(REPO, "phases", "01-instrument", "factworld.pdf")

import markdown  # noqa: E402
from weasyprint import HTML  # noqa: E402

# Times-like body, broad-coverage fallback for ⁰¹ₖ‖∈→≈≥≤±× etc.
SERIF = '"Nimbus Roman", "Liberation Serif", "Noto Serif", "DejaVu Serif", serif'
MONO = '"Nimbus Mono PS", "DejaVu Sans Mono", "Liberation Mono", monospace'

CSS = f"""
@page {{ size: A4; margin: 2.1cm 2.0cm 2.2cm 2.0cm;
         @bottom-center {{ content: counter(page); font-family: {SERIF}; font-size: 9pt; color: #444; }} }}
body {{ font-family: {SERIF}; font-size: 10.5pt; line-height: 1.37; color: #000;
        text-align: justify; hyphens: auto; widows: 2; orphans: 2; }}

h1.title {{ text-align: center; font-size: 16.5pt; line-height: 1.22; font-weight: bold; margin: 0 0 0.2em; }}
.byline {{ text-align: center; font-style: italic; font-size: 11pt; color: #222; margin: 0 0 1.5em; }}
h2 {{ font-size: 12pt; font-weight: bold; margin: 1.35em 0 0.4em; }}
h2[id="abstract"] {{ text-align: center; margin: 0.4em 0 0.45em; }}
h3 {{ font-size: 10.6pt; font-weight: bold; margin: 1.0em 0 0.3em; }}

p {{ margin: 0; text-indent: 1.3em; }}
h1 + p, h2 + p, h3 + p, li > p, td p {{ text-indent: 0; }}
a {{ color: #103a6b; text-decoration: none; overflow-wrap: anywhere; word-break: break-word; }}
strong {{ font-weight: bold; }}

code {{ font-family: {MONO}; font-size: 0.86em; }}
pre {{ font-family: {MONO}; font-size: 8.6pt; line-height: 1.32; background: #f6f6f6;
       padding: 7px 9px; border-radius: 2px; white-space: pre-wrap; text-indent: 0; }}
pre code {{ font-size: 1em; }}

/* booktabs-style tables: horizontal rules only */
table {{ border-collapse: collapse; margin: 0.8em auto; font-size: 9.4pt;
         border-top: 1.1px solid #000; border-bottom: 1.1px solid #000; }}
thead th {{ border-bottom: 0.7px solid #000; font-weight: bold; }}
th, td {{ border: none; padding: 2.5px 13px; text-align: center; }}
td:first-child, th:first-child {{ text-align: left; }}

ul, ol {{ margin: 0.45em 0; }}
li {{ margin: 0.16em 0; }}

/* reference list: no bullets, hanging indent */
h2[id$="references"] + ul {{ list-style: none; padding-left: 0; font-size: 9.3pt; }}
h2[id$="references"] + ul li {{ margin: 0.24em 0; padding-left: 1.6em; text-indent: -1.6em; line-height: 1.3; }}
"""


def split_title(md_text):
    """Pull a leading '**Title**' + '*byline*' block out of the markdown body."""
    lines = md_text.splitlines()
    title, byline, body_start = None, None, 0
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    m = re.match(r"^\*\*(.+?)\*\*\s*$", lines[i]) if i < len(lines) else None
    if m:
        title = m.group(1).strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        mb = re.match(r"^\*(.+?)\*\s*$", lines[j]) if j < len(lines) else None
        if mb:
            byline = mb.group(1).strip()
            body_start = j + 1
        else:
            body_start = i + 1
    return title, byline, "\n".join(lines[body_start:])


def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    out = sys.argv[2] if len(sys.argv) > 2 else (os.path.splitext(src)[0] + ".pdf" if len(sys.argv) > 1 else OUT)
    md_text = open(src, encoding="utf-8").read()
    title, byline, body = split_title(md_text)

    html_body = markdown.markdown(
        body,
        extensions=["tables", "fenced_code", "footnotes", "attr_list", "sane_lists", "smarty", "toc"],
        output_format="html5",
    )
    head = ""
    if title:
        head += f'<h1 class="title">{title}</h1>\n'
    if byline:
        head += f'<div class="byline">{byline}</div>\n'

    html_doc = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{head}{html_body}</body></html>"
    )
    HTML(string=html_doc, base_url=REPO).write_pdf(out)
    print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB)  title={title!r}")


if __name__ == "__main__":
    main()
