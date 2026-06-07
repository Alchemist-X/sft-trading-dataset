"""
ingest/trades.py
================
Load order-book / trade history from a CSV export, group individual fills
into round-trip trades, and compute realized PnL, holding period, and
R-multiple for each completed trade.

EXPECTED CSV COLUMNS:
    timestamp   - ISO 8601 UTC (e.g. 2025-10-12T06:35:00Z)
    symbol      - Trading pair (e.g. BTC/USDT)
    side        - 'buy' or 'sell'
    price       - Fill price (float)
    qty         - Fill quantity in base asset (float)
    fee         - Fee paid in quote currency (float)
    pnl         - Per-fill PnL from exchange (float, 0.0 for opening fills)
    order_type  - 'market' or 'limit'

GROUPING LOGIC:
    - Fills are grouped by symbol.
    - A running net position is maintained per symbol.
    - When net position crosses zero, a round-trip trade is closed.
    - VWAP is used for both entry and exit across multiple partial fills.
    - Fee-inclusive PnL is computed from VWAP entry vs VWAP exit.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures (plain dicts — no mutation after construction)
# ---------------------------------------------------------------------------

# Fill = {timestamp, symbol, side, price, qty, fee, pnl_from_exchange, order_type}
# RoundTrip = {
#   symbol, direction, entry_time, exit_time, avg_entry_price, avg_exit_price,
#   total_qty, total_fee, realized_pnl, realized_pnl_pct, holding_period_hours,
#   r_multiple, verdict, fills_open, fills_close
# }


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 UTC timestamp string to a timezone-aware datetime."""
    ts_str = ts_str.strip().rstrip("Z")
    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp: {ts_str!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_csv(path: Path) -> list[dict]:
    """Load and validate CSV fills from path. Returns list of fill dicts."""
    fills: list[dict] = []
    required_cols = {"timestamp", "symbol", "side", "price", "qty", "fee"}

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no header row.")

            missing = required_cols - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")

            for row_num, row in enumerate(reader, start=2):
                try:
                    fill = {
                        "timestamp": _parse_timestamp(row["timestamp"]),
                        "symbol": row["symbol"].strip().upper(),
                        "side": row["side"].strip().lower(),
                        "price": float(row["price"]),
                        "qty": float(row["qty"]),
                        "fee": float(row.get("fee", 0.0)),
                        "pnl_from_exchange": float(row.get("pnl", 0.0)),
                        "order_type": row.get("order_type", "market").strip().lower(),
                    }
                    if fill["side"] not in ("buy", "sell"):
                        raise ValueError(f"Unknown side: {fill['side']!r}")
                    fills.append(fill)
                except (ValueError, KeyError) as exc:
                    print(f"[ingest/trades] Warning: skipping row {row_num}: {exc}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Trades CSV not found: {path}")

    # Sort chronologically
    fills.sort(key=lambda f: f["timestamp"])
    return fills


def _vwap(prices: list[float], qtys: list[float]) -> float:
    """Compute volume-weighted average price."""
    total_value = sum(p * q for p, q in zip(prices, qtys))
    total_qty = sum(qtys)
    if total_qty == 0.0:
        return 0.0
    return total_value / total_qty


def _compute_r_multiple(
    realized_pnl_pct: float,
    avg_entry: float,
    stop_loss: float | None,
) -> float | None:
    """
    Compute the R-multiple (realized PnL / initial risk).

    If stop_loss is unknown, returns None. The risk-per-share is defined as
    |avg_entry - stop_loss| / avg_entry * 100 (as a percentage).
    """
    if stop_loss is None or avg_entry == 0.0:
        return None
    risk_pct = abs(avg_entry - stop_loss) / avg_entry * 100.0
    if risk_pct == 0.0:
        return None
    return round(realized_pnl_pct / risk_pct, 2)


def _verdict(realized_pnl_pct: float) -> str:
    """Classify trade outcome."""
    if realized_pnl_pct > 1.0:
        return "WIN"
    if realized_pnl_pct < -1.0:
        return "LOSS"
    return "SCRATCH"


def _group_fills_into_round_trips(fills: list[dict]) -> list[dict]:
    """
    Group fills by symbol into round-trip (open → close) trades.

    Uses a position-netting approach:
    - Buy fills increase net position (positive = long).
    - Sell fills decrease net position.
    - When net position reaches 0, a round-trip is completed.

    Returns a list of round-trip trade dicts. Multiple partial fills on
    either side are averaged via VWAP.
    """
    # Group by symbol first
    by_symbol: dict[str, list[dict]] = {}
    for fill in fills:
        by_symbol.setdefault(fill["symbol"], []).append(fill)

    round_trips: list[dict] = []

    for symbol, symbol_fills in by_symbol.items():
        # State per symbol
        net_qty = 0.0
        open_fills: list[dict] = []

        for fill in symbol_fills:
            signed_qty = fill["qty"] if fill["side"] == "buy" else -fill["qty"]
            prev_net = net_qty
            net_qty = round(net_qty + signed_qty, 10)

            if prev_net == 0.0 and net_qty != 0.0:
                # New position opening
                open_fills = [fill]

            elif prev_net != 0.0 and net_qty == 0.0:
                # Position closed — build round-trip
                close_fills = [fill]
                _emit_round_trip(
                    symbol, open_fills, close_fills, round_trips
                )
                open_fills = []

            elif prev_net * net_qty > 0:
                # Same direction — add to open fills
                open_fills.append(fill)

            else:
                # Direction flipped — close current, open opposite
                close_fills = [fill]
                _emit_round_trip(
                    symbol, open_fills, close_fills, round_trips
                )
                open_fills = [fill]

    return round_trips


def _emit_round_trip(
    symbol: str,
    open_fills: list[dict],
    close_fills: list[dict],
    out: list[dict],
) -> None:
    """Compute metrics for a single round-trip and append to out."""
    if not open_fills or not close_fills:
        return

    open_side = open_fills[0]["side"]  # 'buy' = long, 'sell' = short
    direction = "long" if open_side == "buy" else "short"

    open_prices = [f["price"] for f in open_fills]
    open_qtys = [f["qty"] for f in open_fills]
    close_prices = [f["price"] for f in close_fills]
    close_qtys = [f["qty"] for f in close_fills]

    avg_entry = _vwap(open_prices, open_qtys)
    avg_exit = _vwap(close_prices, close_qtys)
    total_qty = sum(open_qtys)
    total_fee = sum(f["fee"] for f in open_fills + close_fills)

    if direction == "long":
        gross_pnl = (avg_exit - avg_entry) * total_qty
    else:
        gross_pnl = (avg_entry - avg_exit) * total_qty

    net_pnl = gross_pnl - total_fee
    cost_basis = avg_entry * total_qty
    realized_pnl_pct = (net_pnl / cost_basis * 100.0) if cost_basis else 0.0

    entry_time = min(f["timestamp"] for f in open_fills)
    exit_time = max(f["timestamp"] for f in close_fills)
    holding_hours = (exit_time - entry_time).total_seconds() / 3600.0

    out.append({
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "avg_entry_price": round(avg_entry, 6),
        "avg_exit_price": round(avg_exit, 6),
        "total_qty": round(total_qty, 8),
        "total_fee": round(total_fee, 6),
        "realized_pnl": round(net_pnl, 4),
        "realized_pnl_pct": round(realized_pnl_pct, 4),
        "holding_period_hours": round(holding_hours, 2),
        "r_multiple": None,  # populated by label.py when stop-loss is known
        "verdict": _verdict(realized_pnl_pct),
        "fills_open": open_fills,
        "fills_close": close_fills,
        "source": "trade_history",
    })


def load_trades_from_csv(path: str | Path) -> Iterator[dict]:
    """
    Load trade history CSV, group fills into round-trip trades, and yield
    one dict per completed trade with computed PnL metrics.

    Args:
        path: Path to the CSV file with trade fill history.

    Yields:
        Round-trip trade dicts with keys:
            symbol, direction, entry_time, exit_time, avg_entry_price,
            avg_exit_price, total_qty, total_fee, realized_pnl,
            realized_pnl_pct, holding_period_hours, r_multiple, verdict,
            fills_open, fills_close, source

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If CSV is malformed or missing required columns.
    """
    path = Path(path)
    fills = _load_csv(path)
    round_trips = _group_fills_into_round_trips(fills)

    for rt in round_trips:
        yield rt
