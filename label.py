"""
label.py
=========
Stage 2: Quality scoring, outcome-based filtering, near-duplicate dedup,
and train/val split assignment.

PIPELINE POSITION:
    ingest/* + transform/to_sft.py --> label.py --> labeled examples

QUALITY SCORE FORMULA:
    Composite score in [0, 1] combining:
      - Outcome verdict  (WIN=1.0, SCRATCH=0.6, LOSS=0.5, missing=0.3)  weight 0.35
      - Source tier      (own_journal=1.0, kol_post=0.7, on_chain=0.6)  weight 0.25
      - Specificity      (has entry + stop + target numeric levels)       weight 0.25
      - Reasoning depth  (word-count proxy, target 50-400 words)         weight 0.15
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

_SOURCE_TIER: dict[str, float] = {
    "own_journal": 1.0,
    "kol_post": 0.7,
    "research_note": 0.75,
    "chat_log": 0.45,
    "on_chain": 0.60,
    "synthetic": 0.20,
}

_VERDICT_SCORE: dict[str, float] = {
    "WIN": 1.0,
    "SCRATCH": 0.6,
    "LOSS": 0.5,
    "EXPIRED": 0.4,
}

_WEIGHTS = {
    "outcome": 0.35,
    "source_tier": 0.25,
    "specificity": 0.25,
    "reasoning_depth": 0.15,
}


def _specificity_score(example: dict) -> float:
    """Score based on how many numeric decision levels are present."""
    decision = example.get("decision", {})
    has_entry = bool(decision.get("entry_zone"))
    has_stop = decision.get("invalidation") is not None
    has_target = bool(decision.get("targets"))
    present = sum([has_entry, has_stop, has_target])
    return present / 3.0


def _reasoning_depth_score(example: dict) -> float:
    """Score based on reasoning word count; penalize very short or very long."""
    text = example.get("reasoning", "")
    words = len(text.split())
    if words < 20:
        return 0.1
    if words < 50:
        return 0.4
    if words <= 400:
        return 1.0
    if words <= 600:
        return 0.8
    return 0.6


def _outcome_score(example: dict) -> float:
    """Score from the outcome verdict field."""
    outcome = example.get("outcome", {})
    if not outcome:
        return 0.3  # no outcome data
    verdict = outcome.get("verdict", "")
    return _VERDICT_SCORE.get(verdict, 0.3)


def compute_quality_score(example: dict) -> float:
    """
    Compute composite quality score in [0, 1] for an SFT example.

    Args:
        example: SFT example dict as produced by transform/to_sft.py.

    Returns:
        Quality score float in [0.0, 1.0].
    """
    source = example.get("source", "synthetic")
    source_score = _SOURCE_TIER.get(source, 0.2)

    score = (
        _WEIGHTS["outcome"] * _outcome_score(example)
        + _WEIGHTS["source_tier"] * source_score
        + _WEIGHTS["specificity"] * _specificity_score(example)
        + _WEIGHTS["reasoning_depth"] * _reasoning_depth_score(example)
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Near-duplicate deduplication
# ---------------------------------------------------------------------------

def _dedup_key(example: dict) -> str:
    """
    Compute a deduplication key for an SFT example.

    Two examples with the same (instrument, date, direction) are considered
    near-duplicates. We hash on these fields for an O(n) exact-dedup pass.
    Semantic embedding-based dedup is out of scope for stdlib-only constraint.
    """
    instrument = (example.get("instrument") or "").upper()
    timestamp = (example.get("timestamp_context") or "")[:10]  # date part only
    direction = (example.get("decision") or {}).get("direction", "")
    source = example.get("source", "")
    # Include source so the same trade from two sources is kept (different value)
    raw = f"{instrument}|{timestamp}|{direction}|{source}"
    return hashlib.md5(raw.encode()).hexdigest()


def deduplicate(examples: list[dict]) -> list[dict]:
    """
    Remove near-duplicate SFT examples using (instrument, date, direction, source) hashing.

    The first occurrence is kept; subsequent duplicates are dropped.

    Args:
        examples: List of SFT example dicts.

    Returns:
        Deduplicated list (preserves order of first occurrence).
    """
    seen: set[str] = set()
    result: list[dict] = []
    dropped = 0

    for ex in examples:
        key = _dedup_key(ex)
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            result.append(ex)

    if dropped:
        print(f"[label] Dedup: removed {dropped} near-duplicate example(s).")
    return result


# ---------------------------------------------------------------------------
# Train / val split (time-ordered)
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> datetime:
    """Parse ISO 8601 timestamp to a UTC datetime."""
    ts = ts.strip().rstrip("Z")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return datetime(2020, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def assign_splits(
    examples: list[dict],
    val_fraction: float = 0.15,
) -> list[dict]:
    """
    Assign train/val splits using a time-ordered cutoff (no random shuffle).

    The most recent `val_fraction` of examples (by timestamp) go to val;
    the rest go to train. There is no held-out test split generated here —
    that is done at training time once a real cutoff date is known.

    Args:
        examples: List of SFT example dicts.
        val_fraction: Fraction of examples to assign to val (default 0.15).

    Returns:
        New list of example dicts with the `split` field updated in-place
        (actually returns new dicts — immutable pattern).
    """
    if not examples:
        return []

    sorted_examples = sorted(
        examples,
        key=lambda e: _parse_ts(e.get("timestamp_context", "")),
    )

    n_val = max(1, int(len(sorted_examples) * val_fraction))
    n_train = len(sorted_examples) - n_val

    result: list[dict] = []
    for i, ex in enumerate(sorted_examples):
        split = "train" if i < n_train else "val"
        result.append({**ex, "split": split})

    return result


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_examples(
    examples: list[dict],
    min_quality: float = 0.0,
    keep_losers: bool = True,
    min_r: float | None = None,
    val_fraction: float = 0.15,
) -> tuple[list[dict], dict]:
    """
    Run the full labeling pipeline on a list of SFT examples.

    Steps:
        1. Compute quality scores.
        2. Deduplicate.
        3. Filter by quality, outcome, and R-multiple thresholds.
        4. Assign time-ordered train/val splits.

    Args:
        examples: Unscored SFT example dicts.
        min_quality: Minimum quality_score for inclusion (0.0 = keep all).
        keep_losers: If False, drop examples with verdict == 'LOSS'.
        min_r: If set, drop examples with r_multiple < min_r (None = no filter).
        val_fraction: Fraction of examples to assign to val split.

    Returns:
        Tuple of (labeled_examples, stats_dict).
    """
    stats: dict[str, Any] = {
        "total_input": len(examples),
        "after_dedup": 0,
        "after_quality_filter": 0,
        "after_outcome_filter": 0,
        "final": 0,
        "train": 0,
        "val": 0,
        "sources": {},
        "verdicts": {},
        "quality_mean": 0.0,
    }

    # 1. Score
    scored = [{**ex, "quality_score": compute_quality_score(ex)} for ex in examples]

    # 2. Deduplicate
    deduped = deduplicate(scored)
    stats["after_dedup"] = len(deduped)

    # 3. Quality filter
    quality_filtered = [
        ex for ex in deduped if ex["quality_score"] >= min_quality
    ]
    stats["after_quality_filter"] = len(quality_filtered)

    # 4. Outcome / R filter
    outcome_filtered: list[dict] = []
    for ex in quality_filtered:
        verdict = (ex.get("outcome") or {}).get("verdict")

        if not keep_losers and verdict == "LOSS":
            continue

        if min_r is not None:
            r = (ex.get("outcome") or {}).get("r_multiple")
            if r is not None and r < min_r:
                continue

        outcome_filtered.append(ex)

    stats["after_outcome_filter"] = len(outcome_filtered)

    # 5. Time-ordered split
    split_examples = assign_splits(outcome_filtered, val_fraction=val_fraction)

    # 6. Collect stats
    stats["final"] = len(split_examples)
    stats["train"] = sum(1 for e in split_examples if e["split"] == "train")
    stats["val"] = sum(1 for e in split_examples if e["split"] == "val")

    source_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    quality_sum = 0.0

    for ex in split_examples:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

        v = (ex.get("outcome") or {}).get("verdict", "none")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

        quality_sum += ex.get("quality_score", 0.0)

    stats["sources"] = source_counts
    stats["verdicts"] = verdict_counts
    stats["quality_mean"] = round(
        quality_sum / len(split_examples) if split_examples else 0.0, 4
    )

    return split_examples, stats
