from __future__ import annotations

import argparse
from datetime import datetime, timezone

from trading_bot.binance_client import BinanceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precharge le cache local des chandeliers Binance.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def parse_date_to_ms(value: str, *, end_of_day: bool = False) -> int:
    date = datetime.strptime(value, "%Y-%m-%d")
    if end_of_day:
        date = date.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(date.replace(tzinfo=timezone.utc).timestamp() * 1000)


def main() -> None:
    args = parse_args()
    client = BinanceClient()
    start_ms = parse_date_to_ms(args.start)
    end_ms = parse_date_to_ms(args.end, end_of_day=True)

    for symbol in args.symbols:
        for interval in args.intervals:
            candles = client.get_klines(symbol=symbol, interval=interval, start_time=start_ms, end_time=end_ms)
            cache_path = client.cache.file_path(symbol, interval)
            print(f"{symbol} {interval}: {len(candles)} bougies en cache -> {cache_path}")


if __name__ == "__main__":
    main()
