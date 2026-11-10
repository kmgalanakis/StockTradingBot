"""One run of the moving average crossover bot.

Command line use:
    python bot.py            # checks every symbol and places orders on signals
    python bot.py --dry-run  # checks every symbol, never places an order

The Docker service in main.py calls run_once() on a schedule.
"""

import argparse
import json
import logging
import os
import sys
import threading
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import strategy
import timezones
import trader
from strategy import BASE_DIR, MARKET_TZ

LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
LAST_RUN_FILE = LOG_DIR / "last_run.json"
STATE_FILE = LOG_DIR / "strategy_state.json"

log = logging.getLogger("bot")


class TimeZoneFormatter(logging.Formatter):
    """Show log times in New York time, plus local time when LOCAL_TIMEZONE is set."""

    def formatTime(self, record, datefmt=None):
        created = datetime.fromtimestamp(record.created, MARKET_TZ)
        return timezones.format_time(created, time_format="%H:%M:%S")


FORMATTER = TimeZoneFormatter("%(asctime)s %(levelname)-7s %(message)s")

# Stops a Telegram /check from running at the same time as the scheduled run.
_run_lock = threading.Lock()


def setup_logging():
    """Log to the console and to logs/bot.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "runs").mkdir(exist_ok=True)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(FORMATTER)
    service_file = RotatingFileHandler(LOG_DIR / "bot.log", maxBytes=5_000_000, backupCount=5)
    service_file.setFormatter(FORMATTER)

    log.setLevel(logging.INFO)
    log.handlers = [console, service_file]
    if timezones.local_tz_error():
        log.warning(timezones.local_tz_error())


def _connect(dry_run):
    """Return (client, positions, symbols with open orders). Dry runs work without Alpaca keys."""
    try:
        client = trader.get_client()
        return client, trader.positions_snapshot(client), trader.open_order_symbols(client)
    except Exception as e:
        if not dry_run:
            raise
        log.warning(f"Could not read Alpaca positions ({e}). Dry run assumes no positions.")
        return None, {}, set()


def _market_closes(config):
    """Download the market fund's prices when the market filter is on, and log its trend."""
    if not strategy.market_filter_on(config):
        return None
    symbol, average = config["market_filter_symbol"], config["market_filter_average"]
    closes = strategy.fetch_prices(symbol, average + 30)
    mean = closes.tail(average).mean()
    trend = "above" if closes.iloc[-1] > mean else "at or below"
    action = "exits all positions" if config["market_filter_exit"] else "blocks new buys"
    log.info(f"Market filter: {symbol} closed at {closes.iloc[-1]:.2f}, {trend} its {average}-day average of {mean:.2f}.")
    if trend != "above":
        log.info(f"Market filter {action} while {symbol} is below its average.")
    return closes


def _high_since_buy(client, symbol, held, ind):
    """Return the highest close since the position was bought, including the buy price."""
    high = held["entry"]
    filled_at = trader.last_buy_fill_time(client, symbol) if client else None
    if filled_at:
        since = ind.close[ind.index.date >= filled_at.astimezone(MARKET_TZ).date()]
        if not since.empty:
            high = max(high, since.max())
    return high


def _buy_quantity(config, price):
    if config["sizing"] == "dollars":
        return float(int(config["trade_dollars"] // price))
    return float(config["trade_size"])


def _free_cash_for_buy(client, config, cost):
    """Sell enough of the idle cash fund to pay for a buy of `cost` dollars and keep the cash reserve."""
    fund = config["idle_cash_symbol"]
    cash, equity = trader.account_cash(client)
    reserve = equity * config["idle_cash_reserve_pct"] / 100
    # 1% extra covers a small price move between the sale and the buy.
    need = cost * 1.01 + reserve - cash
    shares, value = trader.position_value(client, fund)
    if need < 1 or shares <= 0:
        return None
    if need >= value * 0.99:
        message, order_id = trader.place_order(client, fund, "sell", shares)
    else:
        message, order_id = trader.place_dollar_order(client, fund, "sell", need)
    if not trader.wait_for_orders(client, [order_id], timeout=30):
        log.warning(f"The {fund} sale has not filled after 30 seconds. Placing the buy anyway.")
    return message


def _invest_idle_cash(client, config, order_ids, dry_run):
    """Put cash above the reserve into the idle cash fund, or sell the fund if cash went below zero.

    Return a list of order descriptions.
    """
    fund = config["idle_cash_symbol"]
    if order_ids and not trader.wait_for_orders(client, order_ids):
        log.warning(f"Some orders from this run are still open. Idle cash goes into {fund} at the next run.")
        return []
    cash, equity = trader.account_cash(client)
    reserve = equity * config["idle_cash_reserve_pct"] / 100
    extra = cash - reserve
    if extra >= 1:
        side, dollars = "buy", extra
    elif cash < -1:
        _, value = trader.position_value(client, fund)
        side, dollars = "sell", min(-cash, value)
        if dollars < 1:
            return []
    else:
        log.info(f"Idle cash: ${cash:,.2f} in cash, nothing to move.")
        return []
    if dry_run:
        log.info(f"Dry run. Would {side} ${dollars:,.2f} of {fund}. Cash ${cash:,.2f}, reserve ${reserve:,.2f}.")
        return []
    log.info(f"Idle cash: ${cash:,.2f} in cash, reserve ${reserve:,.2f}. Moving ${dollars:,.2f} into {fund}."
             if side == "buy" else f"Idle cash: cash is ${cash:,.2f}. Selling ${dollars:,.2f} of {fund}.")
    message, _ = trader.place_dollar_order(client, fund, side, dollars)
    return [message]


def _load_state():
    """Return (stopped, checked). Kept between runs in logs/strategy_state.json.

    stopped: symbols whose last exit was a trailing stop.
    checked: {symbol: date of the last day the bot decided on}, to find missed crossovers.
    """
    try:
        state = json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        state = {}
    stopped = set(state.get("stopped", []))
    checked = {s: date.fromisoformat(d) for s, d in state.get("checked", {}).items()}
    return stopped, checked


def _save_state(stopped, checked):
    state = {"stopped": sorted(stopped), "checked": {s: d.isoformat() for s, d in sorted(checked.items())}}
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_once(dry_run=False):
    """Check every symbol for a crossover and trade on signals. Return a summary dict."""
    with _run_lock:
        return _run(dry_run)


def _run(dry_run):
    started = datetime.now(MARKET_TZ)

    # Live runs also get their own log file per day, for example logs/runs/2026-10-16.log.
    run_file = None
    if not dry_run:
        run_file = logging.FileHandler(LOG_DIR / "runs" / f"{started.date()}.log")
        run_file.setFormatter(FORMATTER)
        log.addHandler(run_file)

    summary = {
        "started": started.isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "results": {},
        "orders": [],
        "errors": [],
    }

    try:
        log.info(f"========== {'DRY RUN' if dry_run else 'LIVE RUN'} started ==========")
        config = strategy.load_config()
        log.info(
            f"Strategy: {config['short_window']}/{config['long_window']}-day, {config['entry_mode']} mode, "
            f"confirm {config['confirm_days']} day(s), gap {config['min_gap_pct']:g}%, "
            f"trailing stop {config['trailing_stop_pct']:g}%, sizing by {config['sizing']}, "
            + (f"market filter {config['market_filter_symbol']} {config['market_filter_average']}-day"
               f"{' with exit' if config['market_filter_exit'] else ''}"
               if strategy.market_filter_on(config) else "market filter off")
            + (f", idle cash in {config['idle_cash_symbol']}" if strategy.idle_cash_on(config) else "")
        )
        client, positions, pending = _connect(dry_run)
        stopped, checked = _load_state()
        market_closes = _market_closes(config)
        order_ids = []
        idle_cash = strategy.idle_cash_on(config) and client is not None
        if idle_cash and not trader.market_is_open(client):
            log.info(f"Market is closed. Idle cash in {config['idle_cash_symbol']} is left alone until a run while it is open.")
            idle_cash = False
        if idle_cash and config["idle_cash_symbol"] in pending:
            log.info(f"An order for {config['idle_cash_symbol']} is still open. Idle cash is left alone this run.")
            idle_cash = False

        for symbol in config["symbols"]:
            log.info(f"----- {symbol} -----")
            try:
                closes = strategy.fetch_prices(symbol, config["history_days"])
                ind = strategy.compute_indicators(closes, config, market_closes)
                strategy.log_indicators(ind, config)

                held = positions.get(symbol)
                position = {
                    "holding": bool(held),
                    "high": _high_since_buy(client, symbol, held, ind) if held else None,
                    "stopped": symbol in stopped,
                    "missed_buy": strategy.missed_buy(ind, checked.get(symbol)),
                }
                today = ind.iloc[-1]
                if today.bearish:
                    stopped.discard(symbol)
                action, reason = strategy.decide(today, position, config)
            except Exception as e:
                log.warning(f"Skipping {symbol}: {e}")
                summary["results"][symbol] = "error"
                summary["errors"].append(f"{symbol}: {e}")
                continue

            # The day counts as checked once it is handled. After an error, an open
            # order, or a failed order, the next run looks at it again.
            decided_on = ind.index[-1].date()
            if action is None:
                log.info(reason.capitalize() + ".")
                summary["results"][symbol] = reason
                checked[symbol] = decided_on
                continue

            log.info(f"{action.upper()} signal: {reason}.")
            summary["results"][symbol] = f"{action.upper()} ({reason})"
            if symbol in pending:
                log.info("An order for this symbol is still open. Skipping.")
                summary["results"][symbol] += ", order already open"
                continue

            qty = held["qty"] if action == "sell" else _buy_quantity(config, today.close)
            if qty <= 0:
                log.info(f"trade_dollars is below one share at {today.close:.2f}. Skipping.")
                summary["results"][symbol] += ", too expensive for trade_dollars"
                checked[symbol] = decided_on
                continue
            if dry_run:
                log.info(f"Dry run. Would {action} {qty:g} shares. No order placed.")
                continue

            try:
                if idle_cash and action == "buy":
                    sale = _free_cash_for_buy(client, config, qty * today.close)
                    if sale:
                        summary["orders"].append(sale)
                message, order_id = trader.place_order(client, symbol, action, qty)
                summary["orders"].append(message)
                order_ids.append(order_id)
                checked[symbol] = decided_on
                if action == "sell" and reason.startswith("trailing stop"):
                    stopped.add(symbol)
                elif action == "buy":
                    stopped.discard(symbol)
            except Exception as e:
                log.error(f"Order for {symbol} failed: {e}")
                summary["results"][symbol] += ", order failed"
                summary["errors"].append(f"{symbol} order failed: {e}")

        if not dry_run:
            _save_state(stopped, checked)
        if idle_cash:
            try:
                summary["orders"] += _invest_idle_cash(client, config, order_ids, dry_run)
            except Exception as e:
                log.error(f"Moving idle cash into {config['idle_cash_symbol']} failed: {e}")
                summary["errors"].append(f"Idle cash failed: {e}")
    except Exception as e:
        log.exception(f"Run failed: {e}")
        summary["errors"].append(f"Run failed: {e}")
    finally:
        signals = {s: r for s, r in summary["results"].items() if r not in ("no signal", "holding")}
        log.info(f"Checked {len(summary['results'])} symbols. Signals and errors: {signals or 'none'}")
        log.info("========== Run finished ==========")
        if run_file:
            log.removeHandler(run_file)
            run_file.close()

    summary["finished"] = datetime.now(MARKET_TZ).isoformat(timespec="seconds")
    if not dry_run:
        LAST_RUN_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Moving average crossover bot")
    parser.add_argument("--dry-run", action="store_true", help="show signals without placing orders")
    args = parser.parse_args()

    setup_logging()
    run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
