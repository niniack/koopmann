"""Fills report_template.html with the figures (embedded, so the report is one file) and today's time."""

import base64
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
FIGURES = {
    "diagram": ("diagram.png", "Diagram: CLIP blocks 1 to 6 feed the CLS token to the KAE and patch tokens to a control encoder; the KAE replaces blocks 7 to 12."),
    "architecture": ("architecture.png", "Diagram of every layer: state path (encoder, real-start rule, TopK), control path (cross-attention pooling of patch tokens, then an MLP), the eigenvalue step, and the output path (decoder, CLIP's head)."),
    "training": ("training.png", "Line chart of ImageNet accuracy by training step for the KAE with no control, additive control and bilinear control."),
    "lookalikes": ("lookalikes.png", "Six photos: impala, Thomson's gazelle, springbok, kob, sassaby, gerenuk."),
    "species": ("species.png", "Bar chart of species accuracy with CLIP's zero-shot head, before and after tuning eigenvalues."),
    "constant": ("constant.png", "Rows of photos that share a constant mode: object groups without control; snow, motion blur and blurred backgrounds with control."),
    "snow_dial": ("nb/cell9_0.png", "Line chart: accuracy on snowy and normal photos as the snow mode's eigenvalue is turned down."),
    "snow_fixed": ("nb/cell11_1.png", "Eight snowy vehicle photos whose answer is fixed by removing the snow mode."),
    "size_meaning": ("nb/cell15_0.png", "Scatter plot: modes with larger eigenvalue size follow the class label more."),
}

page = (HERE / "report_template.html").read_text()
for key, (file, alt) in FIGURES.items():
    data = base64.b64encode((HERE / file).read_bytes()).decode()
    page = page.replace("{{" + key + "}}", f'<img alt="{alt}" src="data:image/png;base64,{data}">')
now = subprocess.run(["date", "+%-d %B %Y, %H:%M EST"], env={"LC_ALL": "C", "TZ": "America/New_York"}, capture_output=True, text=True).stdout.strip()
(HERE / "report.html").write_text(page.replace("{{updated}}", now))
