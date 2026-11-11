"""
Turns the session transcript (a large .jsonl) into readable text files, one per calendar day (EST), for the review.

Kept: what the user typed, what the assistant wrote, each tool call (shortened) and the start of each tool result.
Dropped: pictures, hidden reasoning, bookkeeping records. Each compaction summary (the long recap written when the
conversation ran out of room) goes to its own file in history/summaries/, with a pointer left in its place.

Usage: extract_history.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = Path("/home/dunstan/.claude/projects/-home-dunstan-git-koopmann/0bba89b8-c8c4-41c4-a0d8-5b331c6750fd.jsonl")
CALL_LIMIT, RESULT_LIMIT = 700, 1_500


def shorten(text, limit):
    text = text.strip()
    return text if len(text) <= limit else f"{text[:limit]} ...[{len(text) - limit} more characters]"


def est(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")) - timedelta(hours=5)


def result_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "[picture]" if part.get("type") == "image" else "") for part in content)


def summary_pointer(record, text):
    """Saves a compaction summary to its own file and returns a pointer to it (None for any other text)."""

    if "continued from a previous conversation" not in text[:300]:
        return None
    path = HERE / "history" / "summaries" / f"summary_{est(record['timestamp']):%m-%d_%H%M}.txt"
    path.write_text(text)
    return f"[compaction summary, saved in full at summaries/{path.name}]"


def render(record, tools):
    """The readable lines of one user or assistant record. `tools` adds the tool calls and results."""

    content = record["message"]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    lines = []
    for part in content:
        if part["type"] == "text" and part["text"].strip():
            who = "ASSISTANT" if record["type"] == "assistant" else "USER"
            lines.append(f"{who}: {summary_pointer(record, part['text']) or shorten(part['text'], 6_000)}")
        elif not tools:
            continue
        elif part["type"] == "tool_use":
            lines.append(f"  CALL {part['name']}: {shorten(json.dumps(part['input'], ensure_ascii=False), CALL_LIMIT)}")
        elif part["type"] == "tool_result":
            lines.append(f"  RESULT: {shorten(result_text(part.get('content') or ''), RESULT_LIMIT)}")
    return lines


days = {"talk": {}, "full": {}}
for number, line in enumerate(SOURCE.open()):
    record = json.loads(line)
    if record.get("type") not in ("user", "assistant") or record.get("isSidechain"):
        continue
    when = est(record["timestamp"])
    for kind in days:
        lines = render(record, kind == "full")
        if lines:
            days[kind].setdefault(when.strftime("%m-%d"), []).append(f"[{when:%m-%d %H:%M} | line {number}]\n" + "\n".join(lines))

for kind, by_day in days.items():
    for day, entries in by_day.items():
        text = "\n\n".join(entries)
        (HERE / "history" / f"{kind}_{day}.txt").write_text(text)
        print(f"{kind}_{day}.txt: {len(entries)} entries, {len(text.splitlines())} lines, about {len(text) // 4_000}k tokens")
