"""
Builds review/report.html: the sections in SECTIONS order inside page.html, with every `{{fig:name}}` replaced by the
picture figs/name.png (embedded, so the report is one file) and a table of contents made from the <h2> headings.

Usage: build.py
"""

import base64
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SECTIONS = ["summary", "model", "speed", "runs", "images", "inside", "meaning", "adaptation", "not_helping", "open", "sources"]


def embed(match):
    name = match.group(1)
    data = base64.b64encode((HERE / "figs" / f"{name}.png").read_bytes()).decode()
    return f'<img alt="Figure: {name.replace("_", " ")}" src="data:image/png;base64,{data}">'


def number_headings(body):
    """Gives each <h2> an id, and returns the body with the list of links to them."""

    titles = re.findall(r"<h2>(.*?)</h2>", body)
    for count, title in enumerate(titles):
        body = body.replace(f"<h2>{title}</h2>", f'<h2 id="s{count}">{title}</h2>', 1)
    links = "".join(f'<li><a href="#s{count}">{re.sub("<.*?>", "", title)}</a></li>' for count, title in enumerate(titles))
    return body, f"<ol>{links}</ol>"


if __name__ == "__main__":
    present = [name for name in SECTIONS if (HERE / "sections" / f"{name}.html").exists()]
    body = "\n".join((HERE / "sections" / f"{name}.html").read_text() for name in present)
    body, contents = number_headings(body)
    body = re.sub(r"\{\{fig:([\w-]+)\}\}", embed, body)
    now = subprocess.run(["date", "+%-d %B %Y, %H:%M EST"], env={"LC_ALL": "C", "TZ": "America/New_York"}, capture_output=True, text=True).stdout.strip()
    page = (HERE / "page.html").read_text().replace("{{body}}", body).replace("{{contents}}", contents).replace("{{updated}}", now)
    (HERE / "report.html").write_text(page)
    print(f"sections: {present}; missing: {[name for name in SECTIONS if name not in present]}; {len(page) / 1e6:.1f} MB; unfilled: {re.findall(r'{{.*?}}', page)[:5]}")
