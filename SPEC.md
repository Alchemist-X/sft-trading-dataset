# SPEC: SFT Trading Reasoning Dataset

**Project:** `sft-trading-dataset`
**Status:** Prototype pipeline implemented — data-collection + labeling pipeline is runnable
**Last updated:** 2026-06-07

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [What a Single Training Example Looks Like](#2-what-a-single-training-example-looks-like)
3. [Proposed JSONL Schema](#3-proposed-jsonl-schema)
4. [Candidate Data Sources](#4-candidate-data-sources)
5. [Data Quality and Deduplication](#5-data-quality-and-deduplication)
6. [Labeling Approach](#6-labeling-approach)
7. [Train / Val / Test Split](#7-train--val--test-split)
8. [SFT vs LoRA and Base Model Considerations](#8-sft-vs-lora-and-base-model-considerations)
9. [Evaluation of the Fine-tuned Model](#9-evaluation-of-the-fine-tuned-model)
10. [Legal / ToS / Licensing Concerns](#10-legal--tos--licensing-concerns)
11. [Phased Build Plan](#11-phased-build-plan)
12. [Existing Repos as Data Sources](#12-existing-repos-as-data-sources)
13. [OPEN QUESTIONS](#13-open-questions)

---

## 1. Problem Statement

Large language models are general-purpose reasoners. When applied to trading, they lack:

- Domain-specific vocabulary and mental models (e.g., "liquidity grab", "funding rate squeeze", "funding arbitrage")
- Grounded market context reasoning (given price action + macro + sentiment → form a thesis)
- Decision-making style calibrated to a specific trader's risk tolerance and edge

**Goal:** Supervised fine-tuning (SFT) of an LLM to produce, given market context, a _reasoned trading thesis and action decision_ that mirrors the style and quality of an expert human trader.

This document specifies the data collection strategy, schema, labeling pipeline, and training considerations. It does **not** produce a trading signal or automated executor — the fine-tuned model is a reasoning assistant.

---

## 2. What a Single Training Example Looks Like

Each example is a three-part structure modeled on chain-of-thought SFT:

```
INSTRUCTION / SYSTEM PROMPT
  → defines the persona, market, and task

CONTEXT (user turn)
  → market snapshot at decision time:
      - asset(s), timeframe, timestamp
      - price, volume, recent OHLCV data
      - relevant macro / news headlines (optional)
      - on-chain data (optional: funding, OI, liquidations)
      - portfolio / position state (optional)

REASONING (model turn — chain-of-thought)
  → step-by-step thesis:
      1. Market structure read
      2. Catalyst / trigger identification
      3. Risk / reward assessment
      4. Conviction and sizing rationale

DECISION / ACTION (conclusion of model turn)
  → structured output:
      - direction: LONG | SHORT | FLAT | WAIT
      - entry_zone: price range or market
      - invalidation: price level that kills the thesis
      - target: price level(s)
      - position_size: % of portfolio or normalized score
      - confidence: 1–5 scale

OUTCOME LABEL (optional, for quality filtering only — NOT in training input)
  → realized_pnl_pct: float
  → holding_period_hours: int
  → verdict: WIN | LOSS | SCRATCH | EXPIRED
```

The outcome label is metadata used during filtering; it is **never** included in the model's input context at inference time (that would be data leakage).

---

## 3. Proposed JSONL Schema

See `schema/sft-example.schema.json` for the JSON Schema definition.

Each line in the exported JSONL is one training example:

```jsonl
{
  "id": "uuid-v4",
  "version": "1.0",
  "split": "train",
  "source": "own_journal",
  "market": "crypto",
  "instrument": "BTC/USDT",
  "timeframe": "4h",
  "timestamp_context": "2025-11-15T08:00:00Z",
  "system_prompt": "You are an expert crypto trader...",
  "context": "BTC is trading at $87,400...",
  "reasoning": "Market structure shows a higher-low formation...",
  "decision": {
    "direction": "LONG",
    "entry_zone": [87000, 87500],
    "invalidation": 85800,
    "targets": [89500, 92000],
    "position_size_pct": 5.0,
    "confidence": 4
  },
  "outcome": {
    "realized_pnl_pct": 2.3,
    "holding_period_hours": 18,
    "verdict": "WIN"
  },
  "quality_score": 0.87,
  "tags": ["breakout", "macro-catalyst", "high-conviction"]
}
```

---

## 4. Candidate Data Sources

### 4.1 Own Trade Journals (HIGHEST QUALITY)

- **What:** Personal written notes made before/during/after trades
- **Format:** Notion pages, markdown files, voice memos, Telegram self-messages
- **Ingestion:** Export Notion DB as JSON/CSV, or scrape local markdown files; transcribe voice with Whisper
- **Labeling:** Match entry timestamp → broker PnL export → outcome label
- **Volume estimate:** Likely 50–500 examples (depends on journal discipline)
- **Quality:** Highest — authentic reasoning in the trader's own voice

### 4.2 KOL Posts / Threads (MEDIUM QUALITY, LEGAL RISK)

- **What:** Public posts from respected analysts on X (Twitter), Substack, Discord, Telegram
- **Format:** Text threads, screenshots, linked charts
- **Ingestion:** Twitter/X API (if still accessible), Nitter scrape, manual copy
- **Labeling:** Outcome must be inferred from market data at the stated entry/target; no guaranteed PnL
- **Volume estimate:** Hundreds to thousands of posts
- **Quality:** Variable — high signal from vetted accounts, noise from the rest
- **Risk:** ToS restrictions, copyright (see Section 10)

### 4.3 Research Notes / Analysis Reports (MEDIUM QUALITY)

- **What:** Personal or team research memos, analyst reports, crypto research newsletters
- **Format:** PDF, Markdown, HTML
- **Ingestion:** PDF extraction (pdfplumber), HTML scraping
- **Labeling:** Outcome-based if a specific trade call is present; otherwise quality = self-consistency score
- **Volume estimate:** Tens to hundreds of documents

### 4.4 Chat Logs (VARIABLE QUALITY)

- **What:** Trading group chats — Telegram, Discord, WhatsApp
- **Format:** Exported JSON (Telegram: Settings → Export), Discord DM/server exports
- **Ingestion:** Parse JSON; filter messages that contain a trade setup (regex + classifier)
- **Labeling:** Difficult — requires linking message timestamp + mentioned instrument → market outcome
- **Volume estimate:** Potentially thousands of raw messages, but low signal density

### 4.5 On-chain Trade History (STRUCTURAL DATA — needs transformation)

- **What:** Wallet-level transaction history: DEX swaps, perp open/close on Hyperliquid, GMX, dYdX
- **Format:** Chain RPC / subgraph / protocol API
- **Ingestion:** Pull via Etherscan API, Hyperliquid REST, or existing `copy-trading-polymarket` pipeline
- **Labeling:** PnL is calculable on-chain — best source of outcome labels
- **Limitation:** No reasoning text — must be paired with journal/chat logs from the same timestamp
- **Volume estimate:** Depends on trading activity

### 4.6 Synthetic / Augmented Data (LAST RESORT)

- **What:** GPT-4 / Claude-generated reasoning traces for historical setups
- **Format:** Generated JSONL
- **Risk:** Model collapse if used as primary training signal; use only to augment
- **Labeling:** Outcome can be computed from historical price data

---

## 5. Data Quality and Deduplication

### Deduplication

- Hash-based dedup on `(instrument, timestamp_context)` tuples
- Semantic similarity dedup using sentence embeddings (e.g., `bge-m3`) with cosine distance threshold ~0.92
- Remove duplicates that are reposts/reshares of the same original analysis

### Quality Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| Outcome verdict | WIN/LOSS/SCRATCH from realized PnL | High |
| Reasoning length | Proxy for depth (target: 150–600 tokens) | Medium |
| Specificity | Contains numeric entry/invalidation/target | High |
| Source tier | Own journal > research note > KOL > chat | High |
| Self-consistency | Does reasoning support the stated decision? | Medium |
| Time to decision | Was it written before or after the fact? | Critical |

### Minimum Bar for Inclusion

An example must:
1. Have a specific instrument and timestamp
2. Have explicit reasoning (not just a bare call)
3. Have an explicit decision direction (LONG/SHORT/FLAT/WAIT)
4. NOT be written post-hoc with outcome knowledge baked in (disqualifying bias)

---

## 6. Labeling Approach

### 6.1 Outcome-Based Filtering (Automated)

For examples with known outcomes:
- **WIN** (PnL > +1%): include with full weight
- **LOSS** (PnL < -1%): include if reasoning quality is high (bad trades with good process are valuable)
- **SCRATCH** (-1% ≤ PnL ≤ +1%): include selectively
- **EXPIRED** (invalidation never triggered, trade not taken): include as WAIT examples

Threshold values TBD pending understanding of the trader's typical R:R.

### 6.2 Self-Consistency Scoring (Automated)

Use a stronger LLM (e.g., Claude Opus) as a judge:
- Prompt: "Given this market context, does the reasoning coherently support the stated decision? Score 1–5."
- Filter out examples with judge score ≤ 2

### 6.3 Human Review (Manual, High Priority)

- All own-journal examples: human review before inclusion
- Random 5–10% sample of other sources: human spot-check
- Create a simple review UI (CSV + label column, or Label Studio)

### 6.4 Negative Examples

Include deliberately bad reasoning (e.g., FOMO chasing without invalidation) labeled as low-quality — useful for preference optimization (DPO) in a later phase.

---

## 7. Train / Val / Test Split

- **Train:** 80% — used for gradient updates
- **Val:** 10% — used for loss monitoring and early stopping
- **Test:** 10% — held out, never seen during training; used for final evaluation

**Important:** Split must be time-ordered, not random. Use a cutoff date:
- All examples before `cutoff_date - N months` → train
- `cutoff_date - N months` to `cutoff_date` → val
- Post `cutoff_date` → test (if live collection continues)

This prevents data leakage from future market knowledge into training.

---

## 8. SFT vs LoRA and Base Model Considerations

### Base Model Options

| Model | Size | Notes |
|-------|------|-------|
| Qwen2.5-7B-Instruct | 7B | Strong reasoning, good Chinese support, efficient |
| Llama-3.1-8B-Instruct | 8B | Broad community support, good base |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Chain-of-thought distilled, strong for reasoning |
| Mistral-7B-Instruct-v0.3 | 7B | Fast, efficient, good instruction following |
| Qwen2.5-14B-Instruct | 14B | Better capacity if data volume is >1K examples |

Base model choice depends heavily on: target language (EN/ZH/bilingual), serving infrastructure, and data volume.

### Training Method

**Phase 1: LoRA / QLoRA (Recommended start)**
- Target rank r=16–64, alpha=32–128
- Apply to Q, K, V, O projections (optionally MLP)
- 4-bit quantization (QLoRA) for memory efficiency on consumer hardware
- Minimal dataset requirements: ~200–500 quality examples for noticeable style shift

**Phase 2: Full SFT (if data volume justifies)**
- Requires >2K high-quality examples
- Requires multi-GPU or cloud training
- Produces a stronger base for further RLHF/DPO

**Phase 3: DPO (if negative examples collected)**
- Pairs (chosen, rejected) for preference learning
- Can fix specific failure modes (hallucinated levels, overconfident calls)

### Training Framework

- `transformers` + `trl` (SFTTrainer / DPOTrainer)
- `unsloth` for 2x+ speedup on LoRA training
- `wandb` for experiment tracking

---

## 9. Evaluation of the Fine-tuned Model

### Automated Metrics

| Metric | Description |
|--------|-------------|
| Perplexity on test set | Standard LM metric; lower is better |
| Decision accuracy | Does model predict correct direction vs. ground truth outcome? |
| Reasoning coherence | LLM-as-judge score on held-out test examples |
| Format compliance | % of outputs that parse correctly into the decision schema |
| Specificity rate | % of outputs that include numeric entry/target/invalidation |

### Behavioral Evaluation

- Vibe eval: compare base model vs. fine-tuned on 10–20 fresh market setups
- A/B blind review: human judge ranks outputs without knowing which model produced them
- Stress test: present setups with contradictory signals; check if model hedges appropriately

### Red Flags to Watch For

- Hallucinated price levels not present in context
- Overconfidence (always high conviction, never WAIT)
- Temporal leakage (referencing events after the context timestamp)
- Style collapse (all outputs sound the same regardless of setup)

---

## 10. Legal / ToS / Licensing Concerns

### Own Data (Safe)

- Personal journals, voice memos, private notes: fully owned, no issues
- Own on-chain transaction history: public data, no ToS issues

### KOL / Public Analyst Content (HIGH RISK)

- **X (Twitter):** ToS prohibits bulk scraping; content is copyright of the author
- **Substack:** ToS prohibits scraping; articles are copyrighted
- **Discord:** ToS prohibits scraping; community content may have its own license
- **Telegram public channels:** Scraping is gray area; content is user-owned
- **Mitigation:** Obtain explicit consent from authors; use only for internal non-commercial model training; do not redistribute the raw data

### Research Reports (MEDIUM RISK)

- Third-party reports: likely copyrighted; internal training use may qualify as fair use (jurisdiction-dependent)
- Consult legal counsel before using licensed research content

### On-chain Data (Safe)

- Blockchain data is public and not copyrightable in most jurisdictions
- Protocol API ToS may restrict bulk access — check per protocol

### Synthetic Data (Safe, with caveats)

- Data generated by commercial LLMs (OpenAI, Anthropic) may have restrictions on use for training competing models — check API ToS
- Using Claude/GPT outputs to fine-tune another model is explicitly restricted by OpenAI's ToS; Anthropic's ToS also restricts this

### General Principle

Prefer data sources you own or have explicit permission to use. Build the training set in tiers: own data first, licensed/consented data second, synthetic augmentation last.

---

## 11. Phased Build Plan

### Phase 0: Scope Lock (RESOLVED — scope clarified 2026-06-07)

- [x] Answer core scope questions (data sources: CSV trade history + KOL tweets)
- [x] Define target instruments (crypto spot/perps, BTC/ETH/SOL focus)
- [x] Identify data sources (Twitter KOL JSON + exchange CSV export)
- [ ] Choose base model (Q5 still open — see Section 13)

### Phase 1: Data Inventory and Schema (COMPLETE)

- [x] Define and finalize JSONL schema (`schema/sft-example.schema.json`)
- [x] Build `ingest/tweets.py` — KOL tweet loader with shared schema + pluggable Twitter fetcher
- [x] Build `ingest/trades.py` — CSV fill loader, round-trip grouping, VWAP PnL computation
- [x] Build `transform/to_sft.py` — chat-format SFT example construction with LLM/template reasoning
- [x] Bundle sample data (`sample/kol_tweets.json`, `sample/trades.csv`)
- [ ] Manual annotation of first 50 examples as gold standard (needs real data)

### Phase 2: Labeling Pipeline (COMPLETE — prototype)

- [x] Quality scoring (composite: outcome + source tier + specificity + reasoning depth)
- [x] Near-duplicate deduplication (hash on instrument × date × direction × source)
- [x] Outcome-based filtering (--keep-losers / --no-keep-losers, --min-r)
- [x] Time-ordered train/val split assignment
- [ ] LLM-judge self-consistency scorer (stub — needs ANTHROPIC_API_KEY integration)
- [ ] Human review workflow (Label Studio or CSV)
- [ ] Target: 200–500 labeled examples with real data

### Phase 3: Export and Training (COMPLETE — prototype)

- [x] `export_jsonl.py` — schema validation, JSONL write, stats Markdown
- [x] `main.py` orchestrator with argparse (`--demo`, `--tweets`, `--trades`, `--min-r`, `--keep-losers`)
- [x] `python3 main.py --demo` produces `out/sft_dataset.jsonl` + `out/sft_dataset.val.jsonl` + `out/stats.md`
- [ ] Set up training environment (Colab Pro / RunPod / local GPU)
- [ ] Train LoRA adapter on chosen base model
- [ ] Log experiments with wandb

### Phase 4: Evaluation and Iteration (ongoing)

- [ ] Run automated evaluation suite
- [ ] Conduct vibe eval / blind A/B review
- [ ] Collect more data based on failure modes
- [ ] Consider DPO phase if negative examples are available

---

## 12. Existing Repos as Data Sources

The following repos in `~/dev-proj/` may be relevant as data sources or infrastructure:

| Repo | Potential Use |
|------|---------------|
| `predict-raven` | May contain signal generation logs or trade predictions — could be mined for reasoning traces |
| `swarm-trading` | May contain multi-agent trading decisions — could provide decision + context pairs |
| `copy-trading-polymarket` | Contains Polymarket copy-trading logic — on-chain trade history is a direct outcome label source |

**This is flagged as an open question** — see Section 13, Q9. Access to these repos and their data structure needs to be confirmed before they can be incorporated into the pipeline.

---

## 13. OPEN QUESTIONS

> Questions marked RESOLVED were clarified 2026-06-07. Remaining questions
> should be answered before Phase 3 (training) begins.

---

**Q1. Which market(s) and instruments? [RESOLVED — PARTIAL]**
- Resolved: Crypto (spot + perps), multi-asset (BTC/ETH/SOL/alts).
- Still open: Which specific exchanges are the primary source of trade CSV exports?

**Q2. What exactly is a "trading idea" (交易思路)? [RESOLVED]**
- A directional thesis (LONG/SHORT a specific instrument) with numeric entry/stop/target.
- WAIT is also a valid decision (e.g. waiting for a liquidity flush before entry).
- Minimum bar: explicit direction + at least one numeric price level.
- Timeframe: primarily swing trades (4h–1d); scalps and position trades also included.

**Q3. What concrete data sources exist? [RESOLVED — PARTIAL]**
- Resolved: Twitter/X KOL posts (shared JSON schema) + exchange CSV trade exports.
- Still open: Which specific KOL accounts to prioritize. Does the trader maintain a written journal?

**Q4. Are outcome (PnL) labels available? [RESOLVED]**
- Resolved: Exchange CSV export with fill-level data provides realized PnL.
- The pipeline groups fills → round-trips → VWAP PnL via `ingest/trades.py`.
- On-chain sources (Hyperliquid, Polymarket) are pluggable but not yet implemented.

**Q5. What is the target base model? [OPEN]**
- Not yet decided. Candidates: Qwen2.5-7B-Instruct (bilingual), DeepSeek-R1-Distill-Qwen-7B.
- Depends on GPU availability and language requirements (EN/ZH bilingual preferred).
- Decision needed before chat template formatting in `transform/to_sft.py` is finalized.

**Q6. What is the expected data volume? [OPEN]**
- Demo pipeline: 10 examples from bundled samples.
- Target for LoRA fine-tuning: 200–500 quality-labeled examples (Phase 2).
- Real volume depends on size of KOL tweet archive and trade history CSV.

**Q7. What is the intended use of the fine-tuned model? [OPEN]**
- Personal trading assistant (local inference) is the assumed use case.
- If deployed as API or published, additional legal review is required.

**Q8. Licensing of KOL content? [OPEN]**
- Bundled sample uses fictional KOL data — no legal issue.
- Real KOL tweets: obtain explicit consent from authors before training.
- For personal internal use only, fair use argument is stronger.

**Q9. Can existing repos be used as data sources? [OPEN]**
- `predict-raven`, `swarm-trading`, `copy-trading-polymarket`: not yet integrated.
- These would be high-value sources if they log decisions + context + outcomes.

**Q10. Acceptable latency / inference cost? [OPEN]**
- Assumed: async analysis (seconds to minutes is acceptable).
- Constrains model size — 7B with QLoRA is likely the right tier.
