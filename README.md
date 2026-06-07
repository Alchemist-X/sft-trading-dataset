# sft-trading-dataset

A data-collection and labeling pipeline for supervised fine-tuning (SFT) of an LLM on trading reasoning (交易思路): given market context, the model produces a reasoned thesis and structured trade decision.

**Status: prototype pipeline implemented — see [SPEC.md](SPEC.md) for full methodology and remaining open questions.**

---

## Quick Start (no API keys required)

```bash
# Clone and run the demo with bundled sample data
python3 main.py --demo
```

Output files appear in `out/`:
- `out/sft_dataset.jsonl` — train split
- `out/sft_dataset.val.jsonl` — val split
- `out/stats.md` — pipeline statistics

---

## Project Structure

```
sft-trading-dataset/
├── main.py                   # Pipeline orchestrator
├── label.py                  # Quality scoring, dedup, train/val split
├── export_jsonl.py           # Schema validation + JSONL writer
│
├── ingest/
│   ├── tweets.py             # KOL tweet loader (shared schema + pluggable fetcher)
│   └── trades.py             # Trade CSV loader (fills → round-trip PnL)
│
├── transform/
│   └── to_sft.py             # Convert trade ideas / trades → SFT chat examples
│
├── sample/
│   ├── kol_tweets.json       # Bundled sample KOL tweets (fictional KOL)
│   └── trades.csv            # Bundled sample trade history
│
├── schema/
│   └── sft-example.schema.json  # JSON Schema for one training example
│
├── pipeline/                 # Legacy scaffolding (original skeletons — superseded)
│   ├── collect.py
│   ├── label.py
│   └── export_jsonl.py
│
├── .env.example              # Environment variable template
└── SPEC.md                   # Full design specification
```

---

## Usage

### Demo mode (bundled sample data, no keys)

```bash
python3 main.py --demo
```

### Custom data

```bash
python3 main.py --tweets path/to/kol_tweets.json --trades path/to/trades.csv
```

### Filtering options

```bash
# Keep only high-R trades (R >= 1.5) and exclude losers
python3 main.py --demo --min-r 1.5 --no-keep-losers

# Minimum quality score threshold
python3 main.py --demo --min-quality 0.5

# Use LLM to expand tweet reasoning (requires ANTHROPIC_API_KEY)
python3 main.py --demo --use-llm
```

### All flags

```
--demo              Run with bundled sample data
--tweets PATH       KOL tweets JSON file (shared schema)
--trades PATH       Trade history CSV file
--out-dir DIR       Output directory (default: ./out)
--min-r FLOAT       Minimum R-multiple filter
--keep-losers       Include LOSS examples (default: True)
--no-keep-losers    Exclude LOSS examples
--min-quality FLOAT Minimum quality_score in [0,1]
--val-fraction FLOAT Fraction for val split (default: 0.15)
--use-llm           Use Claude API to expand tweet reasoning
```

---

## Data Sources

### 1. KOL Tweets (`ingest/tweets.py`)

Loads KOL posts from a JSON file following the **shared tweet schema** (interoperable with the `kol-distill-trader` repo):

```json
{
  "id": "...",
  "author": "KOL Name",
  "handle": "@handle",
  "timestamp": "2025-10-12T06:30:00Z",
  "text": "BTC long at 61200, target 63500, stop 60400",
  "trade": {
    "symbol": "BTC/USDT",
    "direction": "long",
    "entry": 61200,
    "target": 63500,
    "stop": 60400,
    "size": 5.0
  },
  "reasoning": "Liquidity sweep done, CME gap above...",
  "media": []
}
```

**Real Twitter/X fetching:** Set `TWITTER_BEARER_TOKEN` in `.env`. Requires Elevated or Academic API access. See `ingest/tweets.py` (`TwitterFetcherInterface`) for the pluggable interface. **Twitter ToS prohibits bulk scraping — use the v2 API only.**

### 2. Trade History CSV (`ingest/trades.py`)

Loads exchange fill exports with columns:
`timestamp, symbol, side, price, qty, fee, pnl, order_type`

The pipeline groups fills into round-trip trades via position netting and computes VWAP entry/exit, fee-inclusive PnL, holding period, and win/loss verdict.

---

## SFT Example Format

Each JSONL line follows `schema/sft-example.schema.json`. Key fields:

```jsonl
{
  "id": "uuid-v4",
  "version": "1.0",
  "split": "train",
  "source": "kol_post",
  "market": "crypto",
  "instrument": "BTC/USDT",
  "timeframe": "4h",
  "timestamp_context": "2025-10-12T06:30:00Z",
  "system_prompt": "You are a disciplined crypto trader...",
  "context": "Asset: BTC/USDT\nKOL setup posted by...",
  "reasoning": "**Market Structure Read:**\n...",
  "decision": {
    "direction": "LONG",
    "entry_zone": [61200.0],
    "invalidation": 60400.0,
    "targets": [63500.0],
    "position_size_pct": 5.0,
    "confidence": 5
  },
  "outcome": {
    "realized_pnl_pct": 3.55,
    "holding_period_hours": 18.3,
    "verdict": "WIN"
  },
  "quality_score": 0.82,
  "tags": ["bullish-bias", "structured-setup", "liquidity-grab"]
}
```

The `outcome` field is **metadata only** — never included in the model's input at inference time (prevents data leakage).

---

## Quality Scoring

Each example receives a composite `quality_score` in [0, 1]:

| Signal | Weight | Notes |
|--------|--------|-------|
| Outcome verdict (WIN/LOSS/SCRATCH) | 0.35 | WIN=1.0, LOSS=0.5, missing=0.3 |
| Source tier | 0.25 | own_journal=1.0, kol_post=0.7, on_chain=0.6 |
| Specificity (entry/stop/target present) | 0.25 | 3/3 levels = 1.0 |
| Reasoning depth (word count) | 0.15 | Target 50–400 words |

---

## LLM Reasoning Expansion (optional)

If `ANTHROPIC_API_KEY` is set and `--use-llm` is passed, terse tweets are expanded into structured 150-300 word reasoning traces using `claude-haiku-4-5`. Without the key, a deterministic template fallback is used — this always produces valid SFT examples.

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TWITTER_BEARER_TOKEN` | No | Twitter v2 API access for live tweet fetching |
| `ANTHROPIC_API_KEY` | No | Claude API for LLM reasoning expansion |

---

## Legal / ToS Notice

**Twitter/X:** ToS prohibits bulk scraping. Use only the official v2 API with a valid Bearer Token. Content scraped without permission may not be redistributed.

**KOL content copyright:** Posts by third-party analysts are copyrighted by their authors. Obtain explicit consent before using KOL content for commercial model training. Internal non-commercial use may qualify as fair use (jurisdiction-dependent).

**Synthetic data:** Claude-generated content is subject to Anthropic's usage policies. Do not use LLM-generated content to train competing commercial models without reviewing current ToS.

**Own trade history:** Fully safe — you own your trade records.

---

## Interoperability with `kol-distill-trader`

Both `sft-trading-dataset` and `kol-distill-trader` use the same shared tweet schema (`sample/kol_tweets.json` format). The `ingest/tweets.py` module can load files produced by either repo, enabling a shared ingestion pipeline for Twitter KOL data.

---

See [SPEC.md](SPEC.md) for full methodology, open questions, and phased build plan.
