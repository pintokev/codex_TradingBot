from __future__ import annotations

import time

from trading_bot.binance_client import BinanceClient
from trading_bot.runtime_state import RuntimeStateStore
from trading_bot.strategy import TrendFollowingStrategy


def evaluate_latest_signal(
    client: BinanceClient,
    strategy: TrendFollowingStrategy,
    symbol: str,
    interval: str,
    *,
    use_account_position: bool = False,
    force_in_position: bool | None = None,
) -> dict:
    candles = client.get_klines(symbol=symbol, interval=interval, limit=max(400, strategy.runtime_lookback_bars() + 200))
    indicator_frame = strategy.build_indicator_frame(candles)
    strategy.prepare_runtime(
        candles=candles,
        indicator_frame=indicator_frame,
        training_end_index=len(candles) - strategy.config.ml_horizon - 1,
    )
    latest_candle = candles[-1]
    position = None
    in_position = False
    if force_in_position is not None:
        in_position = force_in_position
    elif use_account_position:
        position = client.get_symbol_position(symbol, reference_price=latest_candle.close)
        in_position = bool(position["in_position"])
    signal = strategy.generate_signal(
        candles=candles,
        indicator_frame=indicator_frame,
        index=len(candles) - 1,
        in_position=in_position,
    )
    return {
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "candles": candles,
        "latest_candle": latest_candle,
        "position": position,
    }


def maybe_execute_live_order(
    client: BinanceClient,
    strategy: TrendFollowingStrategy,
    symbol: str,
    interval: str,
    quote_order_qty: float,
    test_order: bool,
) -> dict:
    evaluation = evaluate_latest_signal(client, strategy, symbol, interval, use_account_position=True)
    return execute_signal(
        client=client,
        symbol=symbol,
        quote_order_qty=quote_order_qty,
        test_order=test_order,
        evaluation=evaluation,
    )


def execute_signal(
    client: BinanceClient,
    *,
    symbol: str,
    quote_order_qty: float,
    test_order: bool,
    evaluation: dict,
) -> dict:
    signal = evaluation["signal"]
    position = evaluation["position"] or {}

    if signal.action == "BUY":
        if position.get("in_position"):
            return {
                "executed": False,
                "reason": "Aucun achat execute. Une position est deja ouverte sur le symbole.",
                "signal": signal,
                "latest_candle": evaluation["latest_candle"],
                "position": position,
            }
        response = client.create_market_order(
            symbol=symbol,
            side="BUY",
            quote_order_qty=quote_order_qty,
            test_order=test_order,
        )
        return {
            "executed": True,
            "reason": "Ordre d'achat emis." if not test_order else "Ordre de test Binance valide.",
            "signal": signal,
            "exchange_response": response,
            "latest_candle": evaluation["latest_candle"],
            "position": position,
        }

    if signal.action == "SELL":
        base_asset = position.get("base_asset") or client.get_exchange_info(symbol)["baseAsset"]
        free_balance = float(position.get("free_balance", 0.0))
        if free_balance <= 0:
            return {
                "executed": False,
                "reason": f"Aucune vente executee. Solde {base_asset} nul.",
                "signal": signal,
                "latest_candle": evaluation["latest_candle"],
                "position": position,
            }

        sell_quantity = client.normalize_quantity(symbol, free_balance)
        response = client.create_market_order(
            symbol=symbol,
            side="SELL",
            quantity=sell_quantity,
            test_order=test_order,
        )
        return {
            "executed": True,
            "reason": "Ordre de vente emis." if not test_order else "Ordre de test Binance valide.",
            "signal": signal,
            "exchange_response": response,
            "latest_candle": evaluation["latest_candle"],
            "position": position,
        }

    return {
        "executed": False,
        "reason": f"Aucun ordre execute. Signal courant: {signal.action}.",
        "signal": signal,
        "latest_candle": evaluation["latest_candle"],
        "position": position,
    }


def watch_market(
    client: BinanceClient,
    strategy: TrendFollowingStrategy,
    symbol: str,
    interval: str,
    quote_order_qty: float,
    poll_seconds: int,
    test_order: bool,
    *,
    preset: str,
    max_iterations: int | None = None,
) -> None:
    state_store = RuntimeStateStore()
    state_key = f"{symbol}_{interval}_{preset}"
    iteration = 0

    while True:
        iteration += 1
        evaluation = evaluate_latest_signal(client, strategy, symbol, interval, use_account_position=True)
        signal = evaluation["signal"]
        latest_candle = evaluation["latest_candle"]
        position = evaluation.get("position") or {}
        state = state_store.load(state_key)
        last_executed_candle = state.get("last_executed_candle")

        if latest_candle.open_time == last_executed_candle:
            result = {
                "executed": False,
                "reason": "Aucun ordre execute. Cette bougie a deja ete traitee.",
                "signal": signal,
                "latest_candle": latest_candle,
                "position": position,
            }
        else:
            result = execute_signal(
                client=client,
                symbol=symbol,
                quote_order_qty=quote_order_qty,
                test_order=test_order,
                evaluation=evaluation,
            )

        signal = result["signal"]
        latest_candle = result["latest_candle"]
        position = result.get("position") or {}

        if result["executed"]:
            state_store.save(
                state_key,
                {
                    "last_executed_candle": latest_candle.open_time,
                    "last_action": signal.action,
                    "last_reason": result["reason"],
                },
            )

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{symbol} {interval} candle={latest_candle.open_datetime.isoformat()} "
            f"signal={signal.action} regime={signal.regime} executed={result['executed']} "
            f"position_open={position.get('in_position', False)} reason={result['reason']}"
        )

        if max_iterations is not None and iteration >= max_iterations:
            return
        time.sleep(poll_seconds)
