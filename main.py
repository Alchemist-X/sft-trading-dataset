"""
main.py
=======
Orchestrator for the SFT Trading Dataset pipeline.

USAGE:
    # Demo mode (no API keys required — uses bundled samples):
    python3 main.py --demo

    # Full pipeline with custom data:
    python3 main.py --tweets path/to/kol_tweets.json --trades path/to/trades.csv

    # Filter options:
    python3 main.py --demo --min-r 1.5 --keep-losers
    python3 main.py --demo --min-quality 0.5

    # Use LLM to expand tweet reasoning (requires ANTHROPIC_API_KEY):
    python3 main.py --demo --use-llm

OUTPUT:
    out/sft_dataset.jsonl       — train split
    out/sft_dataset.val.jsonl   — val split
    out/stats.md                — pipeline statistics

PIPELINE:
    1. ingest/tweets.py    — load KOL tweets → trade ideas
    2. ingest/trades.py    — load CSV fills → round-trip trades
    3. transform/to_sft.py — convert to SFT chat-format examples
    4. label.py            — quality score, dedup, outcome filter, split
    5. export_jsonl.py     — validate against schema + write JSONL + stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — allow running from any directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent

def _default_sample(filename: str) -> Path:
    return _ROOT / "sample" / filename


# ---------------------------------------------------------------------------
# Import pipeline modules
# ---------------------------------------------------------------------------
try:
    from ingest.tweets import load_tweets_from_file
    from ingest.trades import load_trades_from_csv
    from transform.to_sft import tweet_idea_to_sft, trade_record_to_sft
    import label as label_mod
    import export_jsonl as export_mod
except ImportError as exc:
    print(f"ERROR: Could not import pipeline module: {exc}")
    print("Make sure you are running from the project root directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SFT Trading Dataset pipeline — build a training dataset from KOL tweets and trade history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py --demo
  python3 main.py --tweets sample/kol_tweets.json --trades sample/trades.csv
  python3 main.py --demo --min-r 1.0 --keep-losers
  python3 main.py --demo --use-llm   # requires ANTHROPIC_API_KEY
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with bundled sample data (no API keys required).",
    )
    parser.add_argument(
        "--tweets",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to KOL tweets JSON file (shared schema). Overrides --demo tweets.",
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to trade history CSV file. Overrides --demo trades.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "out",
        metavar="DIR",
        help="Output directory (default: ./out).",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum R-multiple to include a trade example (e.g. 1.0). "
             "Only applied to examples with a known R value.",
    )
    parser.add_argument(
        "--keep-losers",
        action="store_true",
        default=True,
        help="Include LOSS outcome examples (default: True). "
             "Bad trades with good process are valuable training signal.",
    )
    parser.add_argument(
        "--no-keep-losers",
        action="store_false",
        dest="keep_losers",
        help="Exclude LOSS outcome examples.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Minimum quality_score for inclusion in [0, 1] (default: 0.0 = keep all).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        metavar="FLOAT",
        help="Fraction of examples to assign to val split (default: 0.15).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use Claude API (requires ANTHROPIC_API_KEY) to expand tweet reasoning. "
             "Falls back to deterministic template if key is missing.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _load_tweet_ideas(tweets_path: Path, use_llm: bool) -> list[dict]:
    """Load tweets and convert to SFT examples."""
    ideas: list[dict] = []
    try:
        for idea in load_tweets_from_file(tweets_path):
            sft_example = tweet_idea_to_sft(idea, use_llm=use_llm)
            ideas.append(sft_example)
        print(f"[main] Loaded {len(ideas)} tweet-based SFT example(s) from {tweets_path}")
    except FileNotFoundError as exc:
        print(f"[main] Warning: tweets file not found — {exc}")
    except Exception as exc:
        print(f"[main] Warning: error loading tweets — {exc}")
    return ideas


def _load_trade_examples(trades_path: Path) -> list[dict]:
    """Load trade CSV and convert round-trips to SFT examples."""
    examples: list[dict] = []
    try:
        for trade in load_trades_from_csv(trades_path):
            sft_example = trade_record_to_sft(trade)
            examples.append(sft_example)
        print(f"[main] Loaded {len(examples)} trade-based SFT example(s) from {trades_path}")
    except FileNotFoundError as exc:
        print(f"[main] Warning: trades file not found — {exc}")
    except Exception as exc:
        print(f"[main] Warning: error loading trades — {exc}")
    return examples


def _print_summary(result: dict, stats: dict) -> None:
    """Print a human-readable pipeline summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("SFT DATASET PIPELINE — SUMMARY")
    print(sep)
    print(f"  Total input examples : {stats['total_input']}")
    print(f"  After dedup          : {stats['after_dedup']}")
    print(f"  After quality filter : {stats['after_quality_filter']}")
    print(f"  After outcome filter : {stats['after_outcome_filter']}")
    print(f"  Final examples       : {stats['final']}")
    print(f"    → train            : {stats['train']}")
    print(f"    → val              : {stats['val']}")
    print(f"  Mean quality score   : {stats['quality_mean']:.4f}")
    print()

    verdicts = stats.get("verdicts", {})
    wins = verdicts.get("WIN", 0)
    total_with_outcome = sum(v for k, v in verdicts.items() if k != "none")
    win_rate = (
        f"{wins / total_with_outcome * 100:.1f}%"
        if total_with_outcome > 0
        else "N/A"
    )
    print(f"  Win rate (included)  : {win_rate}")

    sources = stats.get("sources", {})
    if sources:
        print(f"  Source breakdown     : {dict(sorted(sources.items()))}")

    print()
    print(f"  Output files:")
    print(f"    {result['train_path']}  ({result['train_written']} examples)")
    print(f"    {result['val_path']}  ({result['val_written']} examples)")
    print(f"    {result['stats_path']}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Determine input sources
    tweets_path: Path | None = args.tweets
    trades_path: Path | None = args.trades

    if args.demo and tweets_path is None:
        tweets_path = _default_sample("kol_tweets.json")
    if args.demo and trades_path is None:
        trades_path = _default_sample("trades.csv")

    if tweets_path is None and trades_path is None:
        print("ERROR: Provide --demo, --tweets, or --trades (or a combination).")
        parser.print_help()
        sys.exit(1)

    # 1–2. Ingest + transform
    all_examples: list[dict] = []

    if tweets_path is not None:
        all_examples.extend(_load_tweet_ideas(tweets_path, use_llm=args.use_llm))

    if trades_path is not None:
        all_examples.extend(_load_trade_examples(trades_path))

    if not all_examples:
        print("ERROR: No examples were loaded. Check your input files.")
        sys.exit(1)

    # 3. Label (quality score + dedup + filter + split)
    labeled, stats = label_mod.label_examples(
        all_examples,
        min_quality=args.min_quality,
        keep_losers=args.keep_losers,
        min_r=args.min_r,
        val_fraction=args.val_fraction,
    )

    if not labeled:
        print("WARNING: All examples were filtered out. Check filter thresholds.")
        sys.exit(1)

    # 4. Export
    result = export_mod.export_dataset(labeled, stats, args.out_dir)

    # 5. Summary
    _print_summary(result, stats)

    # 6. Print a couple of example JSONL lines for verification
    train_path = result["train_path"]
    print("\nSample JSONL lines from train split:")
    print("-" * 60)
    try:
        with open(train_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 2:
                    break
                obj = json.loads(line)
                # Print a trimmed view
                trimmed = {
                    k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                    for k, v in obj.items()
                    if k not in ("system_prompt",)
                }
                print(json.dumps(trimmed, indent=2, ensure_ascii=False))
                print()
    except Exception as exc:
        print(f"Could not read sample lines: {exc}")


if __name__ == "__main__":
    main()
