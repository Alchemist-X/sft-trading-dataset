"""
pipeline/collect.py
====================
Stage 1: Data Collection

PURPOSE
-------
Ingest raw trading reasoning data from each configured source and normalize it
into intermediate records suitable for the labeling stage (label.py).

This is a SCAFFOLD — no real implementation exists yet. Each TODO marks a
decision point or implementation gap that must be resolved once the open
questions in SPEC.md Section 13 are answered.

INTENDED PIPELINE POSITION
---------------------------
    [raw sources] --> collect.py --> raw_records/*.jsonl --> label.py

OUTPUT FORMAT (intermediate, pre-label)
---------------------------------------
Each output record is a newline-delimited JSON object with at minimum:
    {
        "id": "<uuid>",
        "source": "<source_type>",
        "instrument": "<TBD>",
        "timestamp_context": "<ISO8601>",
        "raw_text": "<unprocessed reasoning text>",
        "decision_raw": "<unprocessed decision string, if extractable>",
        "metadata": {}
    }

SUPPORTED SOURCES (all unimplemented — see TODOs)
--------------------------------------------------
    - own_journal     : Personal trade journal exports (Notion, Markdown, voice memos)
    - kol_post        : KOL/analyst posts scraped from X/Telegram/Discord
    - research_note   : Research PDFs and HTML reports
    - chat_log        : Telegram/Discord group chat JSON exports
    - on_chain        : On-chain trade history from DEX/perp protocols
    - synthetic       : GPT/Claude generated augmentation (last resort)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

# TODO (Q1): Replace with configured instrument list once market/instruments are decided.
# TODO (Q3): Replace with real source paths / API credentials from environment variables.
# TODO (Q4): Decide whether outcome data is fetched here or separately in label.py.

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class SourceCollector:
    """Abstract base for all source collectors.

    Each concrete collector must implement `collect()` which yields raw
    intermediate records as dicts. Collectors must NOT perform any outcome
    labeling or quality scoring — that is the responsibility of label.py.
    """

    source_name: str = "unknown"

    def collect(self) -> Iterator[dict]:
        """Yield normalized raw records from this source.

        Raises:
            NotImplementedError: All subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.collect() not implemented")

    def _make_record(self, raw_text: str, instrument: str, timestamp_context: str, metadata: dict | None = None) -> dict:
        """Construct a standard intermediate record."""
        return {
            "id": str(uuid.uuid4()),
            "source": self.source_name,
            "instrument": instrument,
            "timestamp_context": timestamp_context,
            "raw_text": raw_text,
            "decision_raw": None,  # TODO: extract direction/entry/target via regex or LLM
            "metadata": metadata or {},
        }


# ---------------------------------------------------------------------------
# Source: Own Journal
# ---------------------------------------------------------------------------


class OwnJournalCollector(SourceCollector):
    """Ingest personal trade journal entries.

    Expected input: a directory of Markdown files or a Notion JSON export.
    Each file should represent one or more trade entries.

    TODO (Q3): Confirm journal format (Notion DB export? Local .md files? Voice memos?).
    TODO: Implement Notion API ingestion if journal lives in Notion.
    TODO: Implement Whisper transcription for voice memo files.
    TODO: Parse entry timestamp from file frontmatter or filename.
    TODO: Extract instrument from entry text (regex or LLM-based extraction).
    """

    source_name = "own_journal"

    def __init__(self, journal_dir: str | Path) -> None:
        # TODO (Q3): Validate journal_dir exists and contains expected files.
        self.journal_dir = Path(journal_dir)

    def collect(self) -> Iterator[dict]:
        # TODO: Walk self.journal_dir for .md files.
        # TODO: Parse each file into individual trade entries (split by date header or separator).
        # TODO: For each entry, call self._make_record(...).
        raise NotImplementedError("OwnJournalCollector.collect() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Source: KOL Posts
# ---------------------------------------------------------------------------


class KOLPostCollector(SourceCollector):
    """Ingest public analyst / KOL posts from X (Twitter), Telegram, or Discord.

    LEGAL WARNING: Scraping X and Discord violates their ToS. This collector
    should only be used with content for which explicit permission has been
    obtained, or via official API access. See SPEC.md Section 10.

    TODO (Q3): Confirm which KOL accounts are considered high-signal.
    TODO (Q8): Confirm consent / licensing before implementing any scraping.
    TODO: Implement X API v2 client (requires Elevated access or Academic track).
    TODO: Implement Telegram channel scraper (Telethon or pyrogram).
    TODO: Implement Discord export parser (DiscordChatExporter JSON format).
    TODO: Classify which posts represent a trade setup vs. general commentary.
    """

    source_name = "kol_post"

    def __init__(self, accounts: list[str], platform: str) -> None:
        # TODO (Q3, Q8): Validate platform is one of ['twitter', 'telegram', 'discord'].
        # TODO: Load API credentials from environment variables (never hardcode).
        self.accounts = accounts
        self.platform = platform

    def collect(self) -> Iterator[dict]:
        # TODO: Dispatch to platform-specific scraper based on self.platform.
        # TODO: Filter posts that contain a trade setup (entry, target, invalidation).
        # TODO: Extract instrument and timestamp from post metadata.
        raise NotImplementedError("KOLPostCollector.collect() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Source: Research Notes
# ---------------------------------------------------------------------------


class ResearchNoteCollector(SourceCollector):
    """Ingest research PDFs and HTML reports.

    TODO (Q3): Identify which research subscriptions / note archives are available.
    TODO (Q8): Confirm licensing allows internal training use.
    TODO: Implement PDF text extraction (pdfplumber or pymupdf).
    TODO: Implement HTML scraping with BeautifulSoup or trafilatura.
    TODO: Segment long documents into per-thesis chunks.
    TODO: Extract instrument mentions and date references.
    """

    source_name = "research_note"

    def __init__(self, source_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)

    def collect(self) -> Iterator[dict]:
        # TODO: Walk self.source_dir for .pdf and .html files.
        # TODO: Extract text, segment by section/heading.
        # TODO: Filter segments that represent a trade thesis.
        raise NotImplementedError("ResearchNoteCollector.collect() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Source: Chat Logs
# ---------------------------------------------------------------------------


class ChatLogCollector(SourceCollector):
    """Ingest Telegram or Discord group chat JSON exports.

    Expected input: Telegram Desktop JSON export (Settings > Export Chat History)
    or DiscordChatExporter JSON output.

    TODO (Q3): Confirm which chat groups / channels to include.
    TODO: Implement Telegram JSON parser (result.json structure).
    TODO: Implement DiscordChatExporter JSON parser.
    TODO: Filter messages that contain a trade setup (LLM classifier or regex).
    TODO: Resolve instrument from message text (regex: "BTC", "ETH", ticker symbols).
    TODO: Handle threading / reply chains to preserve context.
    """

    source_name = "chat_log"

    def __init__(self, export_path: str | Path, platform: str = "telegram") -> None:
        # TODO: Validate platform is one of ['telegram', 'discord'].
        self.export_path = Path(export_path)
        self.platform = platform

    def collect(self) -> Iterator[dict]:
        # TODO: Load JSON export from self.export_path.
        # TODO: Iterate messages, classify trade-setup messages.
        # TODO: Group reply chains for context continuity.
        raise NotImplementedError("ChatLogCollector.collect() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Source: On-chain Trade History
# ---------------------------------------------------------------------------


class OnChainCollector(SourceCollector):
    """Ingest on-chain trade history from DEX / perp protocols.

    This source provides outcome labels (entry price, exit price, PnL) but
    NO reasoning text. It must be paired with journal/chat entries from the
    same timestamp window to form complete training examples.

    TODO (Q4): Confirm which wallets and protocols to pull from.
    TODO (Q4): Determine if copy-trading-polymarket repo already has this data.
    TODO: Implement Hyperliquid REST API client for perp trade history.
    TODO: Implement Etherscan/subgraph client for DEX swap history.
    TODO: Compute PnL per trade (entry price, exit price, fees, size).
    TODO: Timestamp-match with journal/chat entries (configurable window).
    """

    source_name = "on_chain"

    def __init__(self, wallet_addresses: list[str], protocols: list[str]) -> None:
        # TODO (Q4): Load wallet addresses from environment variables.
        # TODO: Validate protocols against supported list.
        self.wallet_addresses = wallet_addresses
        self.protocols = protocols

    def collect(self) -> Iterator[dict]:
        # TODO: For each wallet and protocol, fetch trade history via API.
        # TODO: Normalize into intermediate records (no reasoning text — set raw_text="").
        # TODO: Include PnL data in metadata for use by label.py.
        raise NotImplementedError("OnChainCollector.collect() not implemented — see TODOs above")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_collection(output_dir: str | Path) -> None:
    """Run all configured collectors and write intermediate JSONL to output_dir.

    TODO: Load collector configuration from a config file or environment.
    TODO: Run collectors in parallel (concurrent.futures.ThreadPoolExecutor).
    TODO: Write output per source to output_dir/<source_name>.jsonl.
    TODO: Log collection stats (records per source, errors, skipped).
    TODO (Q1, Q2, Q3): Fill in concrete collector instantiation once open questions are answered.
    """
    # TODO: Instantiate collectors based on config.
    # TODO: For each collector, call collect() and write to JSONL.
    raise NotImplementedError("run_collection() not implemented — see TODOs above")


if __name__ == "__main__":
    # TODO: Add argparse for --output-dir, --sources, --config.
    # TODO: Call run_collection() with parsed arguments.
    raise NotImplementedError("CLI entry point not implemented — see TODOs above")
