from __future__ import annotations

import json
import time
from pathlib import Path

from trading_bot.models import Signal
from trading_bot.runtime_state import RuntimeStateStore


class PaperBroker:
    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        portfolio_name: str = "default",
        initial_cash_usdt: float = 10000.0,
        fee_rate: float = 0.001,
    ) -> None:
        self.state_store = RuntimeStateStore(state_dir)
        self.portfolio_key = f"paper_portfolio_{portfolio_name}"
        self.history_path = self.state_store.state_dir / f"paper_history_{portfolio_name}.jsonl"
        self.initial_cash_usdt = initial_cash_usdt
        self.fee_rate = fee_rate

    def load_portfolio(self) -> dict:
        portfolio = self.state_store.load(self.portfolio_key)
        if portfolio:
            return portfolio
        portfolio = {
            "cash_usdt": self.initial_cash_usdt,
            "positions": {},
            "updated_at": int(time.time()),
        }
        self.save_portfolio(portfolio)
        return portfolio

    def save_portfolio(self, portfolio: dict) -> Path:
        portfolio["updated_at"] = int(time.time())
        return self.state_store.save(self.portfolio_key, portfolio)

    def get_position(self, symbol: str) -> dict | None:
        portfolio = self.load_portfolio()
        return portfolio["positions"].get(symbol)

    def in_position(self, symbol: str) -> bool:
        position = self.get_position(symbol)
        return bool(position and position.get("quantity", 0.0) > 0)

    def execute_signal(
        self,
        *,
        symbol: str,
        signal: Signal,
        market_price: float,
        quote_order_qty: float,
        candle_open_time: int,
    ) -> dict:
        portfolio = self.load_portfolio()
        positions = portfolio["positions"]
        current_position = positions.get(symbol)

        if signal.action == "BUY":
            if current_position and current_position.get("quantity", 0.0) > 0:
                return {
                    "executed": False,
                    "reason": "Portefeuille papier deja en position sur ce symbole.",
                    "portfolio": portfolio,
                }

            spend = min(quote_order_qty, float(portfolio["cash_usdt"]))
            if spend <= 0:
                return {
                    "executed": False,
                    "reason": "Cash insuffisant dans le portefeuille papier.",
                    "portfolio": portfolio,
                }

            fee_paid = spend * self.fee_rate
            quantity = (spend - fee_paid) / market_price
            positions[symbol] = {
                "quantity": quantity,
                "avg_entry_price": market_price,
                "opened_at": candle_open_time,
            }
            portfolio["cash_usdt"] -= spend
            self.save_portfolio(portfolio)
            self._append_history(
                {
                    "timestamp": int(time.time()),
                    "candle_open_time": candle_open_time,
                    "symbol": symbol,
                    "action": "BUY",
                    "price": market_price,
                    "quantity": quantity,
                    "gross_notional": spend,
                    "fee_paid": fee_paid,
                    "cash_after": portfolio["cash_usdt"],
                    "reason": signal.reason,
                    "regime": signal.regime,
                }
            )
            return {
                "executed": True,
                "reason": "Achat simule en portefeuille papier.",
                "portfolio": portfolio,
            }

        if signal.action == "SELL":
            if not current_position or current_position.get("quantity", 0.0) <= 0:
                return {
                    "executed": False,
                    "reason": "Aucune position papier a vendre.",
                    "portfolio": portfolio,
                }

            quantity = float(current_position["quantity"])
            gross_notional = quantity * market_price
            fee_paid = gross_notional * self.fee_rate
            net_notional = gross_notional - fee_paid
            portfolio["cash_usdt"] += net_notional
            positions.pop(symbol, None)
            self.save_portfolio(portfolio)
            self._append_history(
                {
                    "timestamp": int(time.time()),
                    "candle_open_time": candle_open_time,
                    "symbol": symbol,
                    "action": "SELL",
                    "price": market_price,
                    "quantity": quantity,
                    "gross_notional": gross_notional,
                    "fee_paid": fee_paid,
                    "cash_after": portfolio["cash_usdt"],
                    "reason": signal.reason,
                    "regime": signal.regime,
                }
            )
            return {
                "executed": True,
                "reason": "Vente simulee en portefeuille papier.",
                "portfolio": portfolio,
            }

        return {
            "executed": False,
            "reason": "Pas d'action a executer en portefeuille papier.",
            "portfolio": portfolio,
        }

    def _append_history(self, payload: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def append_event(self, payload: dict) -> None:
        self._append_history(payload)
