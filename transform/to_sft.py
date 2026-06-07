"""
transform/to_sft.py
====================
Convert structured trade-idea records (from ingest/tweets.py) and
round-trip trade records (from ingest/trades.py) into SFT training
examples conforming to schema/sft-example.schema.json.

CHAT FORMAT:
  - system_prompt : defines the disciplined-trader persona
  - context (user turn) : market snapshot / setup description
  - reasoning + decision (assistant turn) : chain-of-thought + structured call

OUTCOME AS METADATA:
  The realized outcome (PnL / verdict / R-multiple) is attached as the
  `outcome` field for quality filtering ONLY — it is never included in the
  model's input at inference time (that would be data leakage).

LLM EXPANSION:
  If ANTHROPIC_API_KEY is set, a Claude API call expands terse tweets into
  a well-formed multi-step reasoning trace. Otherwise a deterministic template
  is used as fallback. The fallback is always valid and produces usable SFT
  examples.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"

_SYSTEM_PROMPT = (
    "You are a disciplined, process-driven cryptocurrency trader with deep expertise in "
    "technical analysis, on-chain data, and macro market dynamics. "
    "When presented with a market setup, you produce a structured trading thesis that includes: "
    "(1) a clear read of the current market structure, "
    "(2) identification of the key catalyst or trigger, "
    "(3) an explicit risk/reward assessment with numeric entry, target, and invalidation levels, "
    "and (4) a conviction-rated decision (LONG, SHORT, FLAT, or WAIT). "
    "You never trade without a defined invalidation level. "
    "You size positions in proportion to your conviction and account risk tolerance. "
    "Your reasoning is written in English with optional Chinese annotations (中英混用)."
)

_DIRECTION_MAP = {
    "long": "LONG",
    "short": "SHORT",
    "wait": "WAIT",
    "flat": "FLAT",
}


# ---------------------------------------------------------------------------
# Deterministic template helpers
# ---------------------------------------------------------------------------

def _fmt_price(price: float | int | None, symbol: str) -> str:
    """Format a price with appropriate decimal places for the asset."""
    if price is None:
        return "N/A"
    symbol_upper = str(symbol).upper()
    if "BTC" in symbol_upper:
        return f"${price:,.0f}"
    if any(x in symbol_upper for x in ("ETH", "SOL", "BNB", "AVAX")):
        return f"${price:,.1f}"
    return f"${price:,.4f}"


def _build_context_from_tweet(idea: dict) -> str:
    """Build the user-turn context string from a tweet trade idea."""
    symbol = idea["symbol"]
    direction = idea["direction"].upper()
    entry_str = _fmt_price(idea.get("entry"), symbol)
    target_str = _fmt_price(idea.get("target"), symbol)
    stop_str = _fmt_price(idea.get("stop"), symbol)
    size_str = f"{idea['size']}%" if idea.get("size") else "unspecified"
    timestamp = idea["timestamp"]

    lines = [
        f"Asset: {symbol}",
        f"Timestamp: {timestamp}",
        f"KOL setup posted by {idea['author']} ({idea['handle']})",
        f"Proposed direction: {direction}",
        f"Entry zone: {entry_str}",
        f"Target: {target_str}",
        f"Stop / Invalidation: {stop_str}",
        f"Proposed size: {size_str}",
        "",
        f"Tweet text: {idea['text']}",
    ]
    return "\n".join(lines)


def _build_context_from_trade(trade: dict) -> str:
    """Build the user-turn context string from a realized round-trip trade."""
    symbol = trade["symbol"]
    direction = trade["direction"].upper()
    entry_str = _fmt_price(trade.get("avg_entry_price"), symbol)
    exit_str = _fmt_price(trade.get("avg_exit_price"), symbol)
    holding = trade.get("holding_period_hours", 0)

    lines = [
        f"Asset: {symbol}",
        f"Entry timestamp: {trade['entry_time']}",
        f"Exit timestamp: {trade['exit_time']}",
        f"Direction: {direction}",
        f"Average entry price: {entry_str}",
        f"Average exit price: {exit_str}",
        f"Position size (base qty): {trade['total_qty']}",
        f"Holding period: {holding:.1f} hours",
        f"Total fees: ${trade['total_fee']:.2f}",
    ]
    return "\n".join(lines)


def _template_reasoning_from_tweet(idea: dict) -> str:
    """Produce deterministic reasoning for a tweet-sourced trade idea."""
    symbol = idea["symbol"]
    direction = idea["direction"]
    entry = _fmt_price(idea.get("entry"), symbol)
    target = _fmt_price(idea.get("target"), symbol)
    stop = _fmt_price(idea.get("stop"), symbol)
    kol_reasoning = idea.get("reasoning", "").strip()

    parts = [
        f"**Market Structure Read:**",
        f"The setup is on {symbol}, with a proposed {direction.upper()} bias "
        f"at {entry}.",
        "",
        f"**Catalyst / Trigger:**",
        kol_reasoning if kol_reasoning else "See tweet context above for the stated rationale.",
        "",
        f"**Risk / Reward Assessment:**",
    ]

    if idea.get("entry") and idea.get("target") and idea.get("stop"):
        entry_val = float(idea["entry"])
        target_val = float(idea["target"])
        stop_val = float(idea["stop"])
        if direction == "long":
            risk = abs(entry_val - stop_val)
            reward = abs(target_val - entry_val)
        else:
            risk = abs(stop_val - entry_val)
            reward = abs(entry_val - target_val)
        rr = reward / risk if risk > 0 else 0
        parts += [
            f"- Entry: {entry}",
            f"- Target: {target}",
            f"- Stop / Invalidation: {stop}",
            f"- Risk: {_fmt_price(risk, symbol)} | Reward: {_fmt_price(reward, symbol)}",
            f"- R:R ratio: {rr:.2f}:1",
            "",
        ]
    else:
        parts += [
            f"- Entry: {entry}",
            f"- Target: {target}",
            f"- Stop / Invalidation: {stop}",
            "",
        ]

    size = idea.get("size")
    confidence = _infer_confidence(idea)
    parts += [
        f"**Conviction and Sizing:**",
        f"Position size: {size}% of capital." if size else "Position size: not specified.",
        f"Conviction: {confidence}/5 — based on specificity of levels and stated reasoning.",
    ]
    return "\n".join(parts)


def _template_reasoning_from_trade(trade: dict) -> str:
    """Produce deterministic reasoning for a realized trade record."""
    symbol = trade["symbol"]
    direction = trade["direction"].upper()
    entry = _fmt_price(trade.get("avg_entry_price"), symbol)
    exit_p = _fmt_price(trade.get("avg_exit_price"), symbol)
    pnl_pct = trade.get("realized_pnl_pct", 0.0)
    holding = trade.get("holding_period_hours", 0)
    verdict = trade.get("verdict", "UNKNOWN")

    return (
        f"**Market Structure Read:**\n"
        f"Entered a {direction} position on {symbol} at {entry}.\n\n"
        f"**Execution:**\n"
        f"The position was held for {holding:.1f} hours before being closed at {exit_p}.\n\n"
        f"**Risk / Reward:**\n"
        f"Realized PnL: {pnl_pct:+.2f}% (net of fees). Outcome: {verdict}.\n\n"
        f"**Post-trade Reflection:**\n"
        f"This trade is included as a training example to demonstrate "
        f"{'disciplined execution of a profitable setup' if pnl_pct > 0 else 'loss management and process adherence despite an adverse outcome'}."
    )


def _infer_confidence(idea: dict) -> int:
    """Heuristic confidence score 1–5 based on completeness of trade idea."""
    score = 1
    if idea.get("entry"):
        score += 1
    if idea.get("target"):
        score += 1
    if idea.get("stop"):
        score += 1
    if idea.get("reasoning") and len(idea["reasoning"]) > 80:
        score += 1
    return min(score, 5)


def _build_decision_from_tweet(idea: dict) -> dict:
    direction_raw = idea["direction"]
    direction = _DIRECTION_MAP.get(direction_raw.lower(), "WAIT")

    decision: dict[str, Any] = {"direction": direction}

    if idea.get("entry") is not None:
        decision["entry_zone"] = [float(idea["entry"])]
    if idea.get("stop") is not None:
        decision["invalidation"] = float(idea["stop"])
    if idea.get("target") is not None:
        decision["targets"] = [float(idea["target"])]
    if idea.get("size") is not None:
        decision["position_size_pct"] = float(idea["size"])
    decision["confidence"] = _infer_confidence(idea)
    return decision


def _build_decision_from_trade(trade: dict) -> dict:
    direction = _DIRECTION_MAP.get(trade["direction"].lower(), "LONG")
    decision: dict[str, Any] = {
        "direction": direction,
        "entry_zone": [float(trade["avg_entry_price"])],
        "targets": [float(trade["avg_exit_price"])],
        "confidence": 3,
    }
    return decision


def _build_outcome_from_trade(trade: dict) -> dict:
    outcome: dict[str, Any] = {
        "realized_pnl_pct": round(trade.get("realized_pnl_pct", 0.0), 4),
        "holding_period_hours": round(trade.get("holding_period_hours", 0.0), 2),
        "verdict": trade.get("verdict", "SCRATCH"),
        "exit_timestamp": trade.get("exit_time"),
    }
    return {k: v for k, v in outcome.items() if v is not None}


# ---------------------------------------------------------------------------
# LLM expansion (optional, requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_EXPAND_MODEL = "claude-haiku-4-5"  # lightweight — this is called per record


def _expand_reasoning_with_llm(idea: dict, api_key: str) -> str | None:
    """
    Call Claude API to expand a terse tweet into a well-formed reasoning trace.

    Returns expanded reasoning string, or None on failure (caller falls back
    to deterministic template).
    """
    tweet_text = idea.get("text", "")
    kol_reasoning = idea.get("reasoning", "")
    symbol = idea["symbol"]
    direction = idea["direction"].upper()
    entry = _fmt_price(idea.get("entry"), symbol)
    target = _fmt_price(idea.get("target"), symbol)
    stop = _fmt_price(idea.get("stop"), symbol)

    prompt = (
        f"You are a disciplined crypto trader writing a training example for an LLM. "
        f"A KOL posted this trade setup:\n\n"
        f"Symbol: {symbol}, Direction: {direction}, Entry: {entry}, "
        f"Target: {target}, Stop: {stop}\n"
        f"Tweet: {tweet_text}\n"
        f"Their stated reasoning: {kol_reasoning}\n\n"
        f"Expand this into a structured 150-300 word trading reasoning trace with sections: "
        f"1) Market Structure Read, 2) Catalyst/Trigger, 3) Risk/Reward Assessment, "
        f"4) Conviction and Sizing. Be specific, use the numbers above, and be concise."
    )

    payload = {
        "model": _EXPAND_MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ANTHROPIC_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["content"][0]["text"].strip()

    except Exception as exc:
        print(f"[transform/to_sft] LLM expansion failed: {exc}. Using template fallback.")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tweet_idea_to_sft(idea: dict, use_llm: bool = False) -> dict:
    """
    Convert a tweet trade-idea record (from ingest/tweets.py) into an SFT
    training example conforming to schema/sft-example.schema.json.

    Args:
        idea: Trade-idea dict as produced by ingest/tweets.load_tweets_from_file().
        use_llm: If True AND ANTHROPIC_API_KEY is set, call Claude to expand
                 the reasoning trace. Otherwise uses deterministic template.

    Returns:
        SFT example dict (without `split` and `quality_score` — assigned later
        by label.py).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    reasoning = None

    if use_llm and api_key:
        reasoning = _expand_reasoning_with_llm(idea, api_key)

    if not reasoning:
        reasoning = _template_reasoning_from_tweet(idea)

    context = _build_context_from_tweet(idea)
    decision = _build_decision_from_tweet(idea)

    example: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "version": SCHEMA_VERSION,
        "split": "train",  # default; overwritten by label.py splitter
        "source": "kol_post",
        "market": "crypto",
        "instrument": idea["symbol"],
        "timeframe": "4h",
        "timestamp_context": idea["timestamp"],
        "system_prompt": _SYSTEM_PROMPT,
        "context": context,
        "reasoning": reasoning,
        "decision": decision,
        "tags": _auto_tags_from_tweet(idea),
    }
    return example


def trade_record_to_sft(trade: dict) -> dict:
    """
    Convert a round-trip trade record (from ingest/trades.py) into an SFT
    training example conforming to schema/sft-example.schema.json.

    The realized outcome is attached as the `outcome` field (metadata for
    quality filtering — never used as model input at inference time).

    Args:
        trade: Round-trip trade dict as produced by ingest/trades.load_trades_from_csv().

    Returns:
        SFT example dict (without `split` and `quality_score`).
    """
    context = _build_context_from_trade(trade)
    reasoning = _template_reasoning_from_trade(trade)
    decision = _build_decision_from_trade(trade)
    outcome = _build_outcome_from_trade(trade)

    example: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "version": SCHEMA_VERSION,
        "split": "train",
        "source": "on_chain",
        "market": "crypto",
        "instrument": trade["symbol"],
        "timeframe": "4h",
        "timestamp_context": trade["entry_time"],
        "system_prompt": _SYSTEM_PROMPT,
        "context": context,
        "reasoning": reasoning,
        "decision": decision,
        "outcome": outcome,
        "tags": _auto_tags_from_trade(trade),
    }
    return example


def _auto_tags_from_tweet(idea: dict) -> list[str]:
    """Generate taxonomy tags from a tweet trade idea."""
    tags: list[str] = []
    direction = idea.get("direction", "")
    if direction == "long":
        tags.append("bullish-bias")
    elif direction == "short":
        tags.append("bearish-bias")
    if idea.get("has_structured_trade"):
        tags.append("structured-setup")
    if idea.get("reasoning") and len(idea["reasoning"]) > 100:
        tags.append("detailed-reasoning")
    text_lower = idea.get("text", "").lower()
    if "liquidity" in text_lower:
        tags.append("liquidity-grab")
    if "funding" in text_lower:
        tags.append("funding-rate")
    if "breakout" in text_lower:
        tags.append("breakout")
    if "range" in text_lower:
        tags.append("range-trade")
    return tags


def _auto_tags_from_trade(trade: dict) -> list[str]:
    """Generate taxonomy tags from a realized trade record."""
    tags: list[str] = ["realized-trade"]
    verdict = trade.get("verdict", "")
    if verdict == "WIN":
        tags.append("winner")
    elif verdict == "LOSS":
        tags.append("loser")
    elif verdict == "SCRATCH":
        tags.append("scratch")
    hours = trade.get("holding_period_hours", 0)
    if hours < 4:
        tags.append("scalp")
    elif hours < 48:
        tags.append("swing")
    else:
        tags.append("position-trade")
    return tags
