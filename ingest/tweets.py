"""
ingest/tweets.py
================
Load KOL tweets from the shared-schema JSON (bundled sample OR a pluggable
Twitter/X fetcher) and parse each into a structured trade-idea record.

SHARED TWEET SCHEMA (both this repo and kol-distill-trader use this format):
  {
    "id": str,
    "author": str,
    "handle": str,
    "timestamp": str (ISO 8601),
    "text": str,
    "trade": {
      "symbol": str,
      "direction": "long" | "short",
      "entry": float,
      "target": float,
      "stop": float,
      "size": float
    } | null,
    "reasoning": str | null,
    "media": [str]
  }

REAL FETCH NOTE:
  To fetch live tweets from X/Twitter you need a TWITTER_BEARER_TOKEN with
  Elevated or Academic access. Set it in .env (see .env.example).
  Real fetching must respect Twitter ToS — no bulk scraping; use the v2 API
  endpoints only. The pluggable interface is documented below.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Shared schema type alias (documentation only — no runtime enforcement)
# ---------------------------------------------------------------------------

# TweetRecord = dict with keys: id, author, handle, timestamp, text, trade,
#                               reasoning, media

# TradeIdea = dict with keys:
#   tweet_id, author, handle, timestamp, text, symbol, direction, entry,
#   target, stop, size, reasoning, has_structured_trade, source

_DIRECTION_REGEX = re.compile(
    r"\b(long|short|longing|shorting|buy|sell|bullish|bearish)\b",
    re.IGNORECASE,
)
_SYMBOL_REGEX = re.compile(
    r"\b(BTC|ETH|SOL|BNB|XRP|DOGE|AVAX|ARB|OP|MATIC|LINK|UNI|PEPE|WIF)\b(?:/USDT|/USD|-PERP)?",
    re.IGNORECASE,
)
_PRICE_REGEX = re.compile(r"\$?(\d{1,7}(?:[.,]\d{1,4})?)\b")


def _normalize_direction(raw: str) -> str | None:
    """Map raw direction text to 'long' or 'short'."""
    raw_lower = raw.lower()
    if raw_lower in ("long", "longing", "buy", "bullish"):
        return "long"
    if raw_lower in ("short", "shorting", "sell", "bearish"):
        return "short"
    return None


def _extract_direction_from_text(text: str) -> str | None:
    """Best-effort regex extraction of direction from tweet text."""
    match = _DIRECTION_REGEX.search(text)
    if match:
        return _normalize_direction(match.group(1))
    return None


def _extract_symbol_from_text(text: str) -> str | None:
    """Best-effort regex extraction of trading symbol from tweet text."""
    match = _SYMBOL_REGEX.search(text)
    if match:
        symbol = match.group(0).upper()
        # Normalize to BASE/USDT format
        base = symbol.split("/")[0].split("-")[0]
        return f"{base}/USDT"
    return None


def _parse_tweet_to_trade_idea(tweet: dict) -> dict | None:
    """
    Parse a shared-schema tweet dict into a structured trade-idea record.

    Returns None if the tweet has no actionable trade information
    (e.g., it is a pure news repost with no direction).
    """
    try:
        tweet_id = str(tweet["id"])
        author = tweet.get("author", "unknown")
        handle = tweet.get("handle", "")
        timestamp = tweet["timestamp"]
        text = tweet.get("text", "")
        trade_field = tweet.get("trade")  # may be None
        reasoning = tweet.get("reasoning") or ""

        # --- Structured trade field (preferred path) ---
        if trade_field is not None:
            symbol = trade_field.get("symbol", "")
            direction_raw = trade_field.get("direction", "")
            direction = _normalize_direction(direction_raw) or direction_raw
            entry = trade_field.get("entry")
            target = trade_field.get("target")
            stop = trade_field.get("stop")
            size = trade_field.get("size")
            has_structured = True
        else:
            # --- Regex fallback from text ---
            symbol = _extract_symbol_from_text(text) or "UNKNOWN"
            direction = _extract_direction_from_text(text)
            entry = None
            target = None
            stop = None
            size = None
            has_structured = False

        # Reject if no direction could be determined at all
        if direction is None:
            return None

        return {
            "tweet_id": tweet_id,
            "author": author,
            "handle": handle,
            "timestamp": timestamp,
            "text": text,
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "target": target,
            "stop": stop,
            "size": size,
            "reasoning": reasoning,
            "has_structured_trade": has_structured,
            "source": "kol_post",
        }

    except (KeyError, TypeError, ValueError) as exc:
        # Malformed tweet — skip with warning rather than crash
        print(f"[ingest/tweets] Warning: could not parse tweet {tweet.get('id', '?')}: {exc}")
        return None


# ---------------------------------------------------------------------------
# File-based loader (shared-schema JSON)
# ---------------------------------------------------------------------------


def load_tweets_from_file(path: str | Path) -> Iterator[dict]:
    """
    Load KOL tweets from a shared-schema JSON file and yield trade-idea records.

    The JSON file must be a list of tweet objects following the shared schema
    documented at the top of this module.

    Args:
        path: Path to a JSON file containing a list of tweet objects.

    Yields:
        Structured trade-idea dicts for each tweet that contains an actionable
        trade direction.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the root structure is not a list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tweet file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array at root level, got {type(raw).__name__}")

    skipped = 0
    for tweet in raw:
        idea = _parse_tweet_to_trade_idea(tweet)
        if idea is None:
            skipped += 1
            continue
        yield idea

    if skipped:
        print(f"[ingest/tweets] Skipped {skipped} tweet(s) with no actionable trade direction.")


# ---------------------------------------------------------------------------
# Pluggable Twitter/X fetcher interface
# ---------------------------------------------------------------------------

class TwitterFetcherInterface:
    """
    Interface for a real Twitter/X v2 API fetcher.

    IMPLEMENTATION NOTE:
        To implement real fetching:
        1. Set TWITTER_BEARER_TOKEN in your environment (see .env.example).
        2. Subclass this and implement `fetch_user_tweets()`.
        3. Map API response fields to the shared tweet schema.
        4. Respect rate limits: 15 requests / 15 min for user timeline endpoint.
        5. ToS reminder: X prohibits bulk scraping; API access only.

    The returned tweet objects MUST conform to the shared schema so that this
    repo and kol-distill-trader can interoperate on the same data.
    """

    def __init__(self, bearer_token: str | None = None) -> None:
        self._token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not self._token:
            raise EnvironmentError(
                "TWITTER_BEARER_TOKEN is not set. "
                "Set it in your environment or .env file. "
                "Use --demo mode to run without a token."
            )

    def fetch_user_tweets(
        self, handle: str, max_results: int = 100
    ) -> list[dict]:
        """
        Fetch recent tweets for a given @handle via the X v2 API.

        This is a stub. A real implementation would call:
            GET https://api.twitter.com/2/users/by/username/{handle}
        then:
            GET https://api.twitter.com/2/users/{id}/tweets

        Args:
            handle: Twitter handle without the @ symbol.
            max_results: Max tweets to retrieve (1–100 per API call).

        Returns:
            List of tweet objects in the shared schema format.

        Raises:
            NotImplementedError: This stub must be subclassed and implemented.
            urllib.error.HTTPError: On API errors (401 unauthorized, 429 rate limit).
        """
        raise NotImplementedError(
            "Subclass TwitterFetcherInterface and implement fetch_user_tweets(). "
            "See module docstring for implementation notes."
        )

    def _api_get(self, url: str) -> dict:
        """Make an authenticated GET request to the X v2 API."""
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise urllib.error.HTTPError(
                exc.url, exc.code,
                f"Twitter API error {exc.code}: {exc.reason}",
                exc.hdrs, exc.fp,
            ) from exc
