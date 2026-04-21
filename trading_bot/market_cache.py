from __future__ import annotations

import json
import os
from pathlib import Path

from trading_bot.models import Candle

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except Exception:
    pa = None
    feather = None


class MarketDataCache:
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        base_dir = Path(cache_dir or Path.cwd() / ".cache" / "market_data")
        self.cache_dir = base_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, symbol: str, interval: str) -> list[Candle]:
        feather_path = self._file_path(symbol, interval, "feather")
        json_path = self._file_path(symbol, interval, "json")

        if feather_path.exists() and feather is not None:
            table = feather.read_table(feather_path)
            payload = table.to_pylist()
            return [self._dict_to_candle(row) for row in payload]

        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return [self._dict_to_candle(row) for row in payload]

        return []

    def save(self, symbol: str, interval: str, candles: list[Candle]) -> Path:
        payload = [self._candle_to_dict(candle) for candle in candles]

        if feather is not None and pa is not None:
            feather_path = self._file_path(symbol, interval, "feather")
            table = pa.Table.from_pylist(payload)
            feather.write_feather(table, feather_path)
            return feather_path

        json_path = self._file_path(symbol, interval, "json")
        json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return json_path

    def merge(self, symbol: str, interval: str, incoming: list[Candle]) -> tuple[list[Candle], Path]:
        existing = self.load(symbol, interval)
        merged: dict[int, Candle] = {candle.open_time: candle for candle in existing}
        for candle in incoming:
            merged[candle.open_time] = candle
        merged_candles = [merged[key] for key in sorted(merged)]
        path = self.save(symbol, interval, merged_candles)
        return merged_candles, path

    def describe_backend(self) -> str:
        return "feather" if feather is not None and pa is not None else "json"

    def file_path(self, symbol: str, interval: str) -> Path:
        extension = "feather" if feather is not None and pa is not None else "json"
        return self._file_path(symbol, interval, extension)

    def _file_path(self, symbol: str, interval: str, extension: str) -> Path:
        safe_symbol = symbol.upper()
        safe_interval = interval.lower()
        return self.cache_dir / f"{safe_symbol}_{safe_interval}.{extension}"

    @staticmethod
    def _candle_to_dict(candle: Candle) -> dict[str, int | float]:
        return {
            "open_time": candle.open_time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "close_time": candle.close_time,
            "quote_volume": candle.quote_volume,
            "num_trades": candle.num_trades,
            "taker_buy_base_volume": candle.taker_buy_base_volume,
            "taker_buy_quote_volume": candle.taker_buy_quote_volume,
        }

    @staticmethod
    def _dict_to_candle(payload: dict[str, int | float]) -> Candle:
        return Candle(
            open_time=int(payload["open_time"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
            close_time=int(payload["close_time"]),
            quote_volume=float(payload["quote_volume"]),
            num_trades=int(payload["num_trades"]),
            taker_buy_base_volume=float(payload["taker_buy_base_volume"]),
            taker_buy_quote_volume=float(payload["taker_buy_quote_volume"]),
        )


def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"Intervalle non supporte pour le cache: {interval}")


def now_ms() -> int:
    return int(__import__("time").time() * 1000)
