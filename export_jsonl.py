"""
export_jsonl.py
===============
Stage 3: Validate examples against the JSON schema and write JSONL output files.

PIPELINE POSITION:
    label.py --> export_jsonl.py --> out/sft_dataset.jsonl
                                 --> out/sft_dataset.val.jsonl
                                 --> out/stats.md

SCHEMA VALIDATION:
    Each example is validated against schema/sft-example.schema.json using
    a hand-rolled validator (stdlib only — no jsonschema dependency).
    Validation checks required fields, enum values, and type constraints.
    Invalid examples are logged and skipped, not hard-failed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hand-rolled schema validator (stdlib only)
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "id", "version", "split", "source", "instrument",
    "timeframe", "timestamp_context", "system_prompt",
    "context", "reasoning", "decision",
}

_VALID_SPLITS = {"train", "val", "test"}
_VALID_SOURCES = {
    "own_journal", "kol_post", "research_note",
    "chat_log", "on_chain", "synthetic",
}
_VALID_DIRECTIONS = {"LONG", "SHORT", "FLAT", "WAIT"}
_VALID_VERDICTS = {"WIN", "LOSS", "SCRATCH", "EXPIRED"}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"^\d+\.\d+$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)


def validate_example(example: dict) -> list[str]:
    """
    Validate a single SFT example against the schema constraints.

    Returns a list of error messages. An empty list means the example is valid.
    """
    errors: list[str] = []

    # Required fields
    for field in _REQUIRED_FIELDS:
        if field not in example:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors  # stop early if basics are missing

    # id — UUID format
    if not _UUID_RE.match(str(example.get("id", ""))):
        errors.append(f"Field 'id' must be a UUID v4, got: {example.get('id')!r}")

    # version — semver-lite
    if not _VERSION_RE.match(str(example.get("version", ""))):
        errors.append(f"Field 'version' must match N.N pattern, got: {example.get('version')!r}")

    # split
    if example.get("split") not in _VALID_SPLITS:
        errors.append(f"Field 'split' must be one of {sorted(_VALID_SPLITS)}, got: {example.get('split')!r}")

    # source
    if example.get("source") not in _VALID_SOURCES:
        errors.append(f"Field 'source' must be one of {sorted(_VALID_SOURCES)}, got: {example.get('source')!r}")

    # timestamp_context — ISO 8601
    ts = example.get("timestamp_context", "")
    if not _DATETIME_RE.match(str(ts)):
        errors.append(f"Field 'timestamp_context' must be ISO 8601, got: {ts!r}")

    # string fields with minLength
    for field, min_len in [
        ("system_prompt", 10),
        ("context", 20),
        ("reasoning", 50),
        ("instrument", 1),
        ("timeframe", 1),
    ]:
        val = example.get(field, "")
        if not isinstance(val, str):
            errors.append(f"Field '{field}' must be a string.")
        elif len(val) < min_len:
            errors.append(f"Field '{field}' too short (min {min_len} chars), got {len(val)}.")

    # decision object
    decision = example.get("decision", {})
    if not isinstance(decision, dict):
        errors.append("Field 'decision' must be an object.")
    else:
        direction = decision.get("direction")
        if direction not in _VALID_DIRECTIONS:
            errors.append(
                f"Field 'decision.direction' must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got: {direction!r}"
            )
        if "entry_zone" in decision:
            ez = decision["entry_zone"]
            if not isinstance(ez, list) or not (1 <= len(ez) <= 2):
                errors.append("Field 'decision.entry_zone' must be a list of 1–2 numbers.")
        if "invalidation" in decision:
            if not isinstance(decision["invalidation"], (int, float)):
                errors.append("Field 'decision.invalidation' must be a number.")
        if "targets" in decision:
            if not isinstance(decision["targets"], list) or len(decision["targets"]) < 1:
                errors.append("Field 'decision.targets' must be a non-empty list.")
        if "confidence" in decision:
            c = decision["confidence"]
            if not isinstance(c, int) or not (1 <= c <= 5):
                errors.append(f"Field 'decision.confidence' must be int in [1, 5], got: {c!r}")
        if "position_size_pct" in decision:
            ps = decision["position_size_pct"]
            if not isinstance(ps, (int, float)) or not (0 <= ps <= 100):
                errors.append(f"Field 'decision.position_size_pct' must be in [0, 100], got: {ps!r}")

    # outcome (optional)
    outcome = example.get("outcome")
    if outcome is not None:
        if not isinstance(outcome, dict):
            errors.append("Field 'outcome' must be an object or null.")
        else:
            if "verdict" in outcome and outcome["verdict"] not in _VALID_VERDICTS:
                errors.append(
                    f"Field 'outcome.verdict' must be one of {sorted(_VALID_VERDICTS)}, "
                    f"got: {outcome.get('verdict')!r}"
                )
            if "holding_period_hours" in outcome:
                h = outcome["holding_period_hours"]
                if not isinstance(h, (int, float)) or h < 0:
                    errors.append("Field 'outcome.holding_period_hours' must be >= 0.")

    # quality_score (optional)
    qs = example.get("quality_score")
    if qs is not None:
        if not isinstance(qs, (int, float)) or not (0.0 <= qs <= 1.0):
            errors.append(f"Field 'quality_score' must be in [0, 1], got: {qs!r}")

    # tags (optional)
    tags = example.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            errors.append("Field 'tags' must be an array.")

    return errors


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[dict], path: Path) -> int:
    """
    Write a list of SFT example dicts to a JSONL file.

    Each line is one JSON-encoded example. Validates each example first;
    invalid examples are logged and skipped.

    Args:
        examples: List of labeled SFT example dicts.
        path: Output file path.

    Returns:
        Number of examples successfully written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            errors = validate_example(ex)
            if errors:
                ex_id = ex.get("id", "?")
                print(f"[export] Skipping example {ex_id}: {'; '.join(errors)}")
                skipped += 1
                continue
            # Exclude fills from JSONL (internal data — too verbose for training)
            clean = {
                k: v for k, v in ex.items()
                if k not in ("fills_open", "fills_close")
            }
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
            written += 1

    if skipped:
        print(f"[export] Skipped {skipped} invalid example(s).")
    print(f"[export] Wrote {written} example(s) to {path}")
    return written


def write_stats_md(
    stats: dict,
    train_path: Path,
    val_path: Path,
    output_path: Path,
) -> None:
    """
    Write a human-readable stats summary to a Markdown file.

    Args:
        stats: Stats dict as returned by label.label_examples().
        train_path: Path to the written train JSONL (for display).
        val_path: Path to the written val JSONL (for display).
        output_path: Where to write the stats Markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = "\n".join(
        f"| {src} | {count} |"
        for src, count in sorted(stats.get("sources", {}).items())
    )
    verdict_rows = "\n".join(
        f"| {v} | {count} |"
        for v, count in sorted(stats.get("verdicts", {}).items())
    )

    total_wins = stats.get("verdicts", {}).get("WIN", 0)
    total_with_outcome = sum(
        v for k, v in stats.get("verdicts", {}).items() if k != "none"
    )
    win_rate = (
        f"{total_wins / total_with_outcome * 100:.1f}%"
        if total_with_outcome > 0
        else "N/A (no outcome data)"
    )

    md = f"""# SFT Dataset Export Stats

## Pipeline Summary

| Stage | Count |
|-------|-------|
| Total input examples | {stats.get('total_input', 0)} |
| After dedup | {stats.get('after_dedup', 0)} |
| After quality filter | {stats.get('after_quality_filter', 0)} |
| After outcome filter | {stats.get('after_outcome_filter', 0)} |
| **Final (all splits)** | **{stats.get('final', 0)}** |

## Split Breakdown

| Split | Count | File |
|-------|-------|------|
| train | {stats.get('train', 0)} | `{train_path}` |
| val | {stats.get('val', 0)} | `{val_path}` |

## Quality

- Mean quality score: **{stats.get('quality_mean', 0.0):.4f}**
- Win rate (included examples with outcome): **{win_rate}**

## Source Breakdown

| Source | Count |
|--------|-------|
{source_rows}

## Outcome Verdict Distribution

| Verdict | Count |
|---------|-------|
{verdict_rows}

---
*Generated by export_jsonl.py — see SPEC.md for methodology.*
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"[export] Stats written to {output_path}")


def export_dataset(
    labeled: list[dict],
    stats: dict,
    out_dir: Path,
) -> dict:
    """
    Write train JSONL, val JSONL, and stats Markdown to out_dir.

    Args:
        labeled: Labeled and split SFT examples (from label.label_examples()).
        stats: Stats dict from label.label_examples().
        out_dir: Output directory.

    Returns:
        Dict with keys: train_written, val_written, train_path, val_path.
    """
    train_examples = [e for e in labeled if e["split"] == "train"]
    val_examples = [e for e in labeled if e["split"] == "val"]

    train_path = out_dir / "sft_dataset.jsonl"
    val_path = out_dir / "sft_dataset.val.jsonl"
    stats_path = out_dir / "stats.md"

    train_written = write_jsonl(train_examples, train_path)
    val_written = write_jsonl(val_examples, val_path)
    write_stats_md(stats, train_path, val_path, stats_path)

    return {
        "train_written": train_written,
        "val_written": val_written,
        "train_path": train_path,
        "val_path": val_path,
        "stats_path": stats_path,
    }
