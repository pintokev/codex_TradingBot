from __future__ import annotations

import hashlib
import hmac
import os
import time
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

import requests

from trading_bot.market_cache import MarketDataCache, interval_to_milliseconds, now_ms
from trading_bot.models import Candle


class BinanceClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://api.binance.com",
        timeout: int = 30,
        cache_dir: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.cache = MarketDataCache(cache_dir=cache_dir)
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        params = dict(params or {})
        url = f"{self.base_url}{path}"

        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("BINANCE_API_KEY et BINANCE_API_SECRET sont requis pour les requetes signees.")
            params["timestamp"] = int(time.time() * 1000)
            params.setdefault("recvWindow", 5000)
            query = urlencode(params, doseq=True)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature

        response = self.session.request(method, url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        cached = self.cache.load(symbol, interval)
        interval_ms = interval_to_milliseconds(interval)
        requested_end = end_time if end_time is not None else (now_ms() + interval_ms)
        fetch_start = start_time
        fetch_end = end_time

        if cached:
            earliest = cached[0].open_time
            latest = cached[-1].open_time
            needs_prefix = start_time is not None and start_time < earliest
            needs_suffix = requested_end > (latest + interval_ms)

            if not needs_prefix and not needs_suffix:
                return self._slice_candles(cached, start_time=start_time, end_time=end_time, limit=limit)

            if needs_prefix and needs_suffix:
                fresh = self._download_klines(symbol=symbol, interval=interval, start_time=start_time, end_time=end_time, limit=limit)
                merged, _ = self.cache.merge(symbol, interval, fresh)
                return self._slice_candles(merged, start_time=start_time, end_time=end_time, limit=limit)

            if needs_prefix:
                fetch_start = start_time
                fetch_end = earliest - 1
            elif needs_suffix:
                fetch_start = latest + 1
                fetch_end = end_time

        fresh = self._download_klines(symbol=symbol, interval=interval, start_time=fetch_start, end_time=fetch_end, limit=limit)
        merged, _ = self.cache.merge(symbol, interval, fresh)
        return self._slice_candles(merged, start_time=start_time, end_time=end_time, limit=limit)

    def _download_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        candles: list[Candle] = []
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        while True:
            raw_rows = self._request("GET", "/api/v3/klines", params=params, signed=False)
            if not raw_rows:
                break

            for row in raw_rows:
                candles.append(
                    Candle(
                        open_time=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        close_time=int(row[6]),
                        quote_volume=float(row[7]),
                        num_trades=int(row[8]),
                        taker_buy_base_volume=float(row[9]),
                        taker_buy_quote_volume=float(row[10]),
                    )
                )

            if len(raw_rows) < params["limit"]:
                break

            next_start = int(raw_rows[-1][0]) + 1
            if end_time is not None and next_start >= end_time:
                break
            params["startTime"] = next_start

        return candles

    @staticmethod
    def _slice_candles(
        candles: list[Candle],
        *,
        start_time: int | None,
        end_time: int | None,
        limit: int,
    ) -> list[Candle]:
        filtered = candles
        if start_time is not None:
            filtered = [candle for candle in filtered if candle.open_time >= start_time]
        if end_time is not None:
            filtered = [candle for candle in filtered if candle.open_time <= end_time]
        if start_time is None and end_time is None and limit:
            return filtered[-limit:]
        return filtered

    def get_exchange_info(self, symbol: str) -> dict:
        response = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)
        return response["symbols"][0]

    def get_account(self) -> dict:
        return self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)

    def create_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quantity: float | None = None,
        quote_order_qty: float | None = None,
        test_order: bool = False,
    ) -> dict:
        if quantity is None and quote_order_qty is None:
            raise ValueError("quantity ou quote_order_qty est requis.")

        params: dict[str, str | int | float] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
        }
        if quantity is not None:
            params["quantity"] = self._format_decimal(quantity)
        if quote_order_qty is not None:
            params["quoteOrderQty"] = self._format_decimal(quote_order_qty)

        path = "/api/v3/order/test" if test_order else "/api/v3/order"
        return self._request("POST", path, params=params, signed=True)

    def get_asset_balance(self, asset: str) -> float:
        account = self.get_account()
        for balance in account.get("balances", []):
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0

    def get_symbol_position(self, symbol: str, reference_price: float | None = None) -> dict:
        info = self.get_exchange_info(symbol)
        base_asset = info["baseAsset"]
        free_balance = self.get_asset_balance(base_asset)
        min_qty = 0.0
        min_notional = 0.0
        for item in info["filters"]:
            if item["filterType"] == "LOT_SIZE":
                min_qty = float(item["minQty"])
            if item["filterType"] == "NOTIONAL":
                min_notional = float(item.get("minNotional", 0.0))
            if item["filterType"] == "MIN_NOTIONAL" and min_notional == 0.0:
                min_notional = float(item.get("minNotional", 0.0))

        notional = (reference_price or 0.0) * free_balance
        in_position = free_balance >= min_qty and (min_notional == 0.0 or notional >= min_notional * 0.8)
        return {
            "base_asset": base_asset,
            "free_balance": free_balance,
            "min_qty": min_qty,
            "min_notional": min_notional,
            "notional": notional,
            "in_position": in_position,
        }

    def normalize_quantity(self, symbol: str, quantity: float) -> float:
        info = self.get_exchange_info(symbol)
        lot_size = next(filter(lambda item: item["filterType"] == "LOT_SIZE", info["filters"]))
        min_qty = Decimal(lot_size["minQty"])
        step_size = Decimal(lot_size["stepSize"])
        raw_quantity = Decimal(str(quantity))
        normalized = raw_quantity.quantize(step_size, rounding=ROUND_DOWN)
        if normalized < min_qty:
            raise ValueError(f"Quantite trop faible pour {symbol}: {normalized} < {min_qty}")
        return float(normalized)

    @staticmethod
    def _format_decimal(value: float) -> str:
        return format(value, ".8f").rstrip("0").rstrip(".")
