# The review (started 11 Nov 2026, about 11:45 EST)

The user asked (verbatim): "Let's stop for a second, step back, and collect everything we know from the past few days.
Send off agents to go read the previous chat history + training results and compile everything in simple language but
great detail into the report. Please dedicate time into making the graphics in the report easy to understand and clean.
You have several trained models to study. Please dissect each to see where things commonly succeed and where things
commonly fail. Please always keep in mind the three pronged goal: speed + meaning + adaptation. Please make extensive
use of graphs, charts, visual aid."

The three goals, every part of the review is sorted under one of them:
1. **Speed**: a faster stand-in for the real model, with a small allowed loss of accuracy.
2. **Meaning**: the size and the angle of each eigenvalue (and the modes) should mean something.
3. **Adaptation**: fix a domain the real model gets wrong by tuning a few eigenvalues.

## Folders (all under the scratchpad's `review/`)

| Folder / file | What goes there |
|---|---|
| `history/talk_MM-DD.txt` | What the user and the assistant wrote that day (EST), in order. Each entry starts with `[MM-DD HH:MM \| line N]`. |
| `history/full_MM-DD.txt` | The same plus every tool call and the first 1,500 characters of every tool result. Use `grep -n "line N\]"` to jump from a talk entry to the work around it. |
| `history/summaries/` | The recap written each time the conversation ran out of room. Good for orientation; they repeat each other. |
| `digest/` | Each agent's written findings (markdown). |
| `data/` | Numbers behind the charts (csv, json, pt). |
| `figs/` | Finished pictures (png). |
| `sections/` | Draft pieces of the report (html fragments). |
| `common.py` | Shared chart style, colours, model names, `save(fig, name)`, `hidden_images`. **Do not edit it.** |

Scripts an agent writes go in `review/` with a clear name, in the repo's style (short, docstring on top, small functions).
Run them as: `cd <scratchpad>/review && HF_HUB_OFFLINE=1 ../venv/bin/python <script>`.

## Rules for every agent

- **No training, no new runs, no downloads.** This is a review of what exists.
- **Write only inside `review/`.** Read anything in the scratchpad, the repo, the weights cache and wandb. Do not edit
  existing scripts, the old report folder, the repo, or memory. Do not send files or messages to the user. Do not spawn agents.
- **GPU**: only the speed agent uses it. Everyone else sets `CUDA_VISIBLE_DEVICES=""`.
- **CPU cores**: images agent `taskset -c 0-15`, inside agent `taskset -c 16-31`, speed agent `taskset -c 32-47`,
  each with `OMP_NUM_THREADS=16`. Readers need no heavy compute.
- **Never a picture of a bug, arachnid or their relatives.** Any picture of photos must drop the images flagged by
  `common.hidden_images` (or `scratchpad/hidden_mask.pt`, a 50,000-long mask over ImageNet validation), must be looked at
  by the agent photo by photo, and anything doubtful is dropped. Prefer charts to photo galleries.
- **Numbers**: copy them exactly, and next to each write where it comes from (a log path, a wandb run id, or
  `full_MM-DD.txt line N`). If two sources disagree, write both and say so. If a number was later corrected, give the
  corrected one and say what changed. Never fill a gap with a guess; write "not measured".
- **Judging accuracy**: CLIP's zero-shot head on the decoded block-12 state. Never a newly fitted probe.
- **Language**: plain words, short sentences, no made-up terms. Say "mode", "eigenvalue", "size of λ", "angle of λ",
  "control". Define any label at first use. No arrows as connectors in prose.
- **Charts** (dataviz rules): one y-axis; colours from `common.py` in the fixed order BLUE, ORANGE, AQUA, YELLOW, MAGENTA,
  VIOLET; MUTED / INK for reference lines and real CLIP; GOOD / BAD only for right / wrong; label lines and bars directly
  where there are 4 or fewer series, legend otherwise; no number on every point; a title that states the finding
  ("The control is worth 7 points"), not the topic; axis labels in words; model names from `common.MODELS`.
  Open every picture you make with the Read tool, check for overlaps, cut-off text and unreadable sizes, and fix it.
- **Draft sections** (`sections/<name>.html`): plain html fragments using only `<h2> <h3> <p> <ul> <li> <table> <tr> <th>
  <td> <b> <code>` and `<p class="note">`. Put a figure as `{{fig:<file name without .png>}}` on its own line, followed by
  one `<p class="note">` saying how to read it. Lead each section with the finding in one or two sentences.
- **Final reply to the lead**: at most 25 lines: the files written, the 5 to 10 most important findings with numbers, and
  anything that is uncertain or that contradicts what the lead believes. The detail belongs in the files.

## Models with checkpoints (weights cache `tokae/`)

`common.MODELS` lists the eleven main ones. Loaders that exist: `angle_tests.load_kae` (cls_*, share_*),
`animals/probe_kaes.py` (`load_kae`, `get_output` for on_*), `second_analysis/demo_utils.py` and `scripts/tokae/train.py`
(older im_* models on stored features). Validation tensors: `control_train.get_val(None, "ViT-B-32", device)`.
Other checkpoints of interest: `im_iso_2x4k_s1` (second seed), `im_iso_2x4k_wd`, `im_realstart_2x4k_wd`, `_s1`,
`im_iso_2x8k`, `im_dino_w21_wd`, `im_dino_w21_wd_e30` (DINOv3).

## Provisional outline of the report

0. One-page summary: the model, the three goals, a scorecard, what stands and what fell.
1. The model and how it changed over the five days (timeline).
2. Speed: accuracy against cost; what closed the gap; what did not.
3. Where the KAEs succeed and where they fail (the dissection).
4. Meaning: size of λ, angle of λ, constant modes, the control problem.
5. Adaptation: snow, look-alike species, DINOv3.
6. Tried, not helping. 7. Open problems and decisions. 8. Every run in one table; sources.
