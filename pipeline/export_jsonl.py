"""
pipeline/export_jsonl.py
=========================
Stage 3: Export and Train/Val/Test Split

PURPOSE
-------
Take fully labeled records from label.py, apply final quality filtering,
perform a time-ordered train/val/test split, validate each record against
the JSON schema, and write the final JSONL files for model training.

This is a SCAFFOLD — no real implementation exists yet. Each TODO marks a
decision point or implementation gap that must be resolved once the open
questions in SPEC.md Section 13 are answered.

INTENDED PIPELINE POSITION
---------------------------
    label.py --> labeled_records/*.jsonl --> export_jsonl.py
                                            --> output/train.jsonl
                                            --> output/val.jsonl
                                            --> output/test.jsonl
                                            --> output/dataset_card.json

KEY DESIGN DECISIONS (all TBD — see OPEN QUESTIONS)
----------------------------------------------------
    - Minimum quality_score threshold for inclusion (e.g., 0.5)
    - Cutoff date for train/val/test boundary (Q6 — depends on data horizon)
    - Whether to include LOSS outcome examples (see SPEC.md Section 6.1)
    - Output format: raw JSONL vs. chat-template formatted (depends on base model, Q5)
    - Token budget per example (depends on base model context window, Q5)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# TODO (Q5): Import the correct chat template formatter once base model is chosen.
# TODO: Import jsonschema for schema validation against schema/sft-example.schema.json.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for the export stage.

    All defaults are placeholders — must be tuned once open questions are answered.
    """

    min_quality_score: float = 0.5
    """Minimum quality_score for a record to be included in any split."""

    include_loss_examples: bool = True
    """Whether to include examples with outcome.verdict == LOSS.
    Rationale: bad trades with good process are valuable (see SPEC.md Section 6.1).
    TODO (Q2, Q4): Confirm with the trader whether loss examples should be included.
    """

    train_cutoff_date: str = "TBD"
    """ISO 8601 date string. Records with timestamp_context before this date go to train.
    TODO (Q6): Set based on data collection horizon and desired test recency.
    """

    val_cutoff_date: str = "TBD"
    """ISO 8601 date string. Records between train_cutoff_date and val_cutoff_date go to val.
    TODO (Q6): Set once train_cutoff_date is determined.
    """

    max_tokens_per_example: int = 4096
    """Maximum token count for context + reasoning + decision combined.
    TODO (Q5): Set based on chosen base model's context window and training memory budget.
    """

    output_format: str = "raw"
    """Output format for JSONL records.
    Options:
        'raw'          : Records conforming directly to sft-example.schema.json.
        'chat_template': Records formatted with the base model's chat template
                         (e.g., Qwen's <|im_start|>/<|im_end|> tokens,
                          Llama's [INST]/[/INST] tokens).
    TODO (Q5): Set once base model is chosen.
    """


# ---------------------------------------------------------------------------
# Quality Filter
# ---------------------------------------------------------------------------


class QualityFilter:
    """Apply quality thresholds and business rules to select training examples.

    TODO: Implement each filter criterion.
    TODO: Log filter stats (how many records dropped per criterion).
    """

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def passes(self, record: dict) -> tuple[bool, str]:
        """Return (passes, reason_if_rejected) for a labeled record.

        Checks applied in order:
            1. quality_score >= config.min_quality_score
            2. Record has required fields (instrument, timestamp_context, reasoning, decision)
            3. decision.direction is not None
            4. If not config.include_loss_examples: reject LOSS outcome records
            5. Token length <= config.max_tokens_per_example

        TODO: Implement all checks above.
        TODO (Q2): Add instrument whitelist check once target instruments are confirmed.
        TODO (Q5): Implement token counting using the base model's tokenizer.
        """
        raise NotImplementedError("QualityFilter.passes() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class Deduplicator:
    """Remove duplicate records before splitting.

    Two-pass deduplication:
        1. Exact dedup: hash on (instrument, timestamp_context, direction)
        2. Semantic dedup: embed reasoning text and remove near-duplicates
           with cosine similarity > threshold.

    TODO: Implement exact hash-based dedup.
    TODO: Implement semantic dedup using sentence-transformers (bge-m3 or similar).
    TODO: Tune semantic similarity threshold (default: 0.92).
    TODO: Log how many records were removed and why.
    """

    def __init__(self, semantic_threshold: float = 0.92) -> None:
        self.semantic_threshold = semantic_threshold
        # TODO: Load embedding model lazily to avoid startup cost if not needed.

    def deduplicate(self, records: list[dict]) -> list[dict]:
        """Return deduplicated list of records.

        TODO: Run exact dedup pass first, then semantic dedup on the remainder.
        """
        raise NotImplementedError("Deduplicator.deduplicate() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Train/Val/Test Splitter
# ---------------------------------------------------------------------------


class TimeOrderedSplitter:
    """Split records into train/val/test using time-ordered cutoff dates.

    CRITICAL: This must be a temporal split, NOT a random split. Using a
    random split would leak future market context into the training set
    (survivorship bias and look-ahead bias). See SPEC.md Section 7.

    TODO (Q6): Implement once cutoff dates are configured in ExportConfig.
    TODO: Validate that test split contains records more recent than val,
          and val more recent than train.
    TODO: Log split statistics (count per split, date range per split).
    """

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def split(self, records: list[dict]) -> dict[str, list[dict]]:
        """Return {'train': [...], 'val': [...], 'test': [...]} dict.

        TODO: Parse timestamp_context for each record.
        TODO: Assign split based on cutoff dates.
        TODO: Set record['split'] field for each record.
        """
        raise NotImplementedError("TimeOrderedSplitter.split() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate each record against schema/sft-example.schema.json before writing.

    TODO: Load schema from schema/sft-example.schema.json.
    TODO: Use jsonschema.validate() to check each record.
    TODO: Collect validation errors and report as a summary (don't silently skip).
    TODO: Decide whether invalid records are dropped or cause a hard failure.
    """

    def __init__(self, schema_path: str | Path) -> None:
        self.schema_path = Path(schema_path)
        # TODO: Load and parse schema JSON at init time.

    def validate(self, record: dict) -> list[str]:
        """Return list of validation error messages (empty list = valid).

        TODO: Run jsonschema validation and collect error messages.
        """
        raise NotImplementedError("SchemaValidator.validate() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Chat Template Formatter
# ---------------------------------------------------------------------------


class ChatTemplateFormatter:
    """Format records into the base model's chat template for direct training.

    Different base models use different special tokens:
        - Qwen2.5:  <|im_start|>system\\n...\\n<|im_end|> etc.
        - Llama 3:  <|begin_of_text|><|start_header_id|>system<|end_header_id|> etc.
        - Mistral:  [INST] ... [/INST]

    TODO (Q5): Implement once base model is chosen.
    TODO: Use the model's tokenizer.apply_chat_template() where available.
    TODO: Ensure system_prompt, context, and reasoning+decision are mapped to
          correct roles (system, user, assistant).
    """

    def __init__(self, model_family: str) -> None:
        # TODO (Q5): Validate model_family against supported list.
        self.model_family = model_family

    def format(self, record: dict) -> dict:
        """Return record with added 'formatted_text' field containing the chat template.

        TODO: Map record fields to chat template turns.
        TODO: Validate token count stays within max_tokens_per_example.
        """
        raise NotImplementedError("ChatTemplateFormatter.format() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Dataset Card Generator
# ---------------------------------------------------------------------------


def generate_dataset_card(splits: dict[str, list[dict]], output_path: str | Path) -> None:
    """Write a JSON dataset card summarizing the exported dataset.

    Output fields:
        - total_examples, per_split counts
        - source distribution
        - instrument distribution
        - date range
        - quality_score statistics (mean, std, percentiles)
        - outcome verdict distribution
        - schema_version

    TODO: Implement stat collection across all splits.
    TODO: Write to output_path/dataset_card.json.
    """
    raise NotImplementedError("generate_dataset_card() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_export(labeled_records_dir: str | Path, schema_path: str | Path, output_dir: str | Path, config: ExportConfig | None = None) -> None:
    """Run the full export pipeline.

    Steps:
        1. Load all labeled records from labeled_records_dir.
        2. Apply quality filter (QualityFilter).
        3. Deduplicate (Deduplicator).
        4. Split into train/val/test (TimeOrderedSplitter).
        5. Validate each record (SchemaValidator).
        6. Optionally format with chat template (ChatTemplateFormatter).
        7. Write train.jsonl, val.jsonl, test.jsonl to output_dir.
        8. Write dataset_card.json to output_dir.

    TODO (Q5, Q6): Finalize ExportConfig defaults once open questions answered.
    TODO: Add progress logging with record counts at each stage.
    TODO: Make output_dir and schema_path configurable via environment variables.
    """
    if config is None:
        config = ExportConfig()

    raise NotImplementedError("run_export() not implemented — see TODOs above")


if __name__ == "__main__":
    # TODO: Add argparse for --labeled-records-dir, --schema-path, --output-dir, --config.
    # TODO: Call run_export() with parsed arguments.
    raise NotImplementedError("CLI entry point not implemented — see TODOs above")
