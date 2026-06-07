"""
pipeline/label.py
==================
Stage 2: Labeling and Quality Scoring

PURPOSE
-------
Take raw intermediate records from collect.py and produce fully labeled,
quality-scored records ready for export. Three labeling strategies are applied
in sequence:

    1. Outcome-based labeling  — match trade entry timestamp + instrument to
                                  realized PnL from broker export or on-chain data.
    2. Self-consistency scoring — use an LLM-as-judge to score whether the
                                  reasoning coherently supports the decision.
    3. Human review flagging    — flag records for human review based on
                                  configurable criteria; produce a review queue.

This is a SCAFFOLD — no real implementation exists yet. Each TODO marks a
decision point or implementation gap that must be resolved once the open
questions in SPEC.md Section 13 are answered.

INTENDED PIPELINE POSITION
---------------------------
    collect.py --> raw_records/*.jsonl --> label.py --> labeled_records/*.jsonl
                                                         --> review_queue.csv

INPUT FORMAT
------------
    Intermediate records as produced by collect.py (see collect.py docstring).

OUTPUT FORMAT
-------------
    Fully populated records conforming to schema/sft-example.schema.json,
    with `outcome` and `quality_score` fields populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# TODO (Q4): Confirm PnL data source (broker CSV export? on-chain API? copy-trading-polymarket?).
# TODO (Q5): Confirm which LLM to use as judge (Claude API? local model?).
# TODO: Load API keys from environment variables — never hardcode.


# ---------------------------------------------------------------------------
# Outcome Labeler
# ---------------------------------------------------------------------------


@dataclass
class OutcomeLabel:
    """Container for a realized trade outcome."""

    realized_pnl_pct: float | None = None
    holding_period_hours: float | None = None
    verdict: str | None = None  # WIN | LOSS | SCRATCH | EXPIRED
    exit_timestamp: str | None = None


class OutcomeLabeler:
    """Match raw records to realized PnL outcomes.

    Strategy:
        1. Load a PnL source (broker CSV or on-chain trade history).
        2. For each raw record with a known instrument and timestamp_context,
           search for a matching trade entry within a configurable time window.
        3. If a match is found, compute verdict from PnL thresholds.
        4. If no match is found, attempt to compute theoretical PnL from
           historical OHLCV data using the entry_zone and target/invalidation.

    TODO (Q4): Decide primary PnL data source and implement loader.
    TODO (Q4): Implement broker CSV parser (format TBD by exchange).
    TODO (Q4): Implement on-chain PnL fetcher (reuse or import from copy-trading-polymarket?).
    TODO: Implement OHLCV-based theoretical PnL calculator for unmatched records.
    TODO: Tune WIN/LOSS/SCRATCH thresholds to match the trader's typical R:R (Q2).
    TODO: Handle multi-leg trades and partial fills.
    """

    def __init__(self, pnl_source_path: str | Path, win_threshold_pct: float = 1.0, loss_threshold_pct: float = -1.0) -> None:
        # TODO (Q4): Load PnL data from pnl_source_path.
        self.pnl_source_path = Path(pnl_source_path)
        self.win_threshold_pct = win_threshold_pct
        self.loss_threshold_pct = loss_threshold_pct

    def label(self, record: dict) -> OutcomeLabel:
        """Look up and return the outcome for a given raw record.

        TODO: Implement timestamp + instrument matching against PnL source.
        TODO: Return OutcomeLabel with all fields populated if match found.
        TODO: Return OutcomeLabel with verdict=EXPIRED if no match and time window exceeded.
        """
        raise NotImplementedError("OutcomeLabeler.label() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Self-Consistency Scorer (LLM-as-Judge)
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyScore:
    """Container for the LLM judge's self-consistency evaluation."""

    score: int | None = None  # 1–5
    rationale: str | None = None
    model_used: str | None = None


class SelfConsistencyScorer:
    """Use a strong LLM as a judge to score reasoning coherence.

    Prompt the judge with:
        - The market context
        - The reasoning trace
        - The stated decision

    Ask: "Does the reasoning coherently support the decision given the context?
    Score 1 (incoherent) to 5 (highly coherent). Explain briefly."

    TODO (Q5): Decide which judge model to use (Claude API, GPT-4, or local).
    TODO: Implement Anthropic API client call (use claude-opus-4 or claude-sonnet-4-5).
    TODO: Implement prompt template with few-shot examples for calibration.
    TODO: Implement retry logic with exponential backoff for API rate limits.
    TODO: Cache judge responses to avoid redundant API calls (hash on record id).
    TODO: Consider running judge in parallel for throughput.
    """

    def __init__(self, judge_model: str, api_key_env_var: str = "ANTHROPIC_API_KEY") -> None:
        # TODO (Q5): Validate judge_model against supported models.
        # TODO: Load API key from environment — never hardcode.
        self.judge_model = judge_model
        self.api_key_env_var = api_key_env_var

    def score(self, record: dict) -> ConsistencyScore:
        """Call the judge LLM and return a self-consistency score.

        TODO: Build prompt from record['context'], record['reasoning'], record['decision_raw'].
        TODO: Parse judge response to extract integer score and rationale.
        TODO: Handle parsing failures gracefully (default to score=None, flag for review).
        """
        raise NotImplementedError("SelfConsistencyScorer.score() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Quality Score Aggregator
# ---------------------------------------------------------------------------


class QualityScoreAggregator:
    """Combine multiple quality signals into a single composite score in [0, 1].

    Signals and default weights (TBD — adjust after seeing real data distribution):
        - Outcome verdict (WIN=1.0, SCRATCH=0.6, LOSS=0.5, EXPIRED=0.4, None=0.3) : weight 0.35
        - Self-consistency score (normalized to [0, 1])                             : weight 0.30
        - Source tier (own_journal=1.0, research=0.8, kol=0.6, chat=0.4, synth=0.2): weight 0.20
        - Specificity (has numeric entry + invalidation + target)                   : weight 0.15

    TODO: Tune weights based on correlation with human review verdicts.
    TODO: Add reasoning length signal (penalize very short or very long traces).
    TODO: Add temporal authenticity signal (written before vs. after outcome known).
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        # TODO: Validate weights sum to 1.0.
        self.weights = weights or {
            "outcome": 0.35,
            "consistency": 0.30,
            "source_tier": 0.20,
            "specificity": 0.15,
        }

    def compute(self, record: dict, outcome: OutcomeLabel, consistency: ConsistencyScore) -> float:
        """Return composite quality score in [0, 1].

        TODO: Implement each sub-score computation.
        TODO: Apply weights and return weighted sum.
        """
        raise NotImplementedError("QualityScoreAggregator.compute() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Human Review Queue
# ---------------------------------------------------------------------------


class ReviewQueueBuilder:
    """Flag records for human review and export to CSV.

    Records are flagged when:
        - They are from the own_journal source (always review)
        - Their quality_score is between 0.4 and 0.7 (borderline)
        - Their self-consistency score is None (judge failed to score)
        - They are randomly sampled at configured rate (e.g., 5%)

    TODO: Implement CSV exporter (columns: id, source, instrument, timestamp,
          reasoning_excerpt, decision_raw, outcome_verdict, quality_score, flag_reason).
    TODO: Integrate with Label Studio for a proper review UI (optional).
    TODO: Implement review result ingestion (read human verdicts back into labeled records).
    """

    def __init__(self, output_path: str | Path, always_review_sources: list[str] | None = None, sample_rate: float = 0.05) -> None:
        self.output_path = Path(output_path)
        self.always_review_sources = always_review_sources or ["own_journal"]
        self.sample_rate = sample_rate

    def should_review(self, record: dict, quality_score: float) -> tuple[bool, str]:
        """Return (should_review, reason) for a given record and quality score.

        TODO: Implement all flagging criteria listed in the class docstring.
        """
        raise NotImplementedError("ReviewQueueBuilder.should_review() not implemented — see TODOs above")

    def export(self, records_to_review: list[dict]) -> None:
        """Write flagged records to CSV at self.output_path.

        TODO: Implement CSV writer with configured columns.
        """
        raise NotImplementedError("ReviewQueueBuilder.export() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_labeling(raw_records_dir: str | Path, output_dir: str | Path, review_queue_path: str | Path) -> None:
    """Run the full labeling pipeline on all raw intermediate records.

    TODO: Load all raw JSONL files from raw_records_dir.
    TODO: Instantiate OutcomeLabeler, SelfConsistencyScorer, QualityScoreAggregator, ReviewQueueBuilder.
    TODO: For each record: label outcome, score consistency, compute quality, flag for review.
    TODO: Write fully labeled records to output_dir/labeled_records.jsonl.
    TODO: Write review queue to review_queue_path.
    TODO: Log stats: total records, label coverage, score distribution, review queue size.
    TODO (Q4): Wire up correct PnL source path from config/env.
    TODO (Q5): Wire up correct judge model from config/env.
    """
    raise NotImplementedError("run_labeling() not implemented — see TODOs above")


if __name__ == "__main__":
    # TODO: Add argparse for --raw-records-dir, --output-dir, --review-queue, --config.
    # TODO: Call run_labeling() with parsed arguments.
    raise NotImplementedError("CLI entry point not implemented — see TODOs above")
