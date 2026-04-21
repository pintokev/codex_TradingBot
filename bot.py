from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from trading_bot.backtest import Backtester
from trading_bot.binance_client import BinanceClient
from trading_bot.live import evaluate_latest_signal, maybe_execute_live_order, watch_market
from trading_bot.paper import PaperBroker
from trading_bot.runtime_state import RuntimeStateStore
from trading_bot.strategy import TrendFollowingStrategy, build_risk_config, build_strategy_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot de trading Binance axe tendance.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Execute un backtest sur Binance.")
    backtest.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    backtest.add_argument("--preset", choices=["trend_1d", "intraday_1h", "hybrid_1h", "hybrid_ml_1h"], default="trend_1d")
    backtest.add_argument("--interval")
    backtest.add_argument("--start", default="2024-01-01")
    backtest.add_argument("--end", default="2026-04-20")
    backtest.add_argument("--capital", type=float, default=10000.0)
    backtest.add_argument("--fee-rate", type=float)
    backtest.add_argument("--stop-loss", type=float)
    backtest.add_argument("--take-profit", type=float)
    backtest.add_argument("--position-fraction", type=float)

    signal = subparsers.add_parser("signal", help="Affiche le dernier signal.")
    signal.add_argument("--symbol", required=True)
    signal.add_argument("--preset", choices=["trend_1d", "intraday_1h", "hybrid_1h", "hybrid_ml_1h"], default="trend_1d")
    signal.add_argument("--interval")

    live = subparsers.add_parser("live", help="Execute un achat si le signal est haussier.")
    live.add_argument("--symbol", required=True)
    live.add_argument("--preset", choices=["trend_1d", "intraday_1h", "hybrid_1h", "hybrid_ml_1h"], default="trend_1d")
    live.add_argument("--interval")
    live.add_argument("--quote-order-qty", type=float, required=True)
    live.add_argument("--test-order", action="store_true")

    watch = subparsers.add_parser("watch", help="Sonde le marche en boucle et execute les ordres si necessaire.")
    watch.add_argument("--symbol", required=True)
    watch.add_argument("--preset", choices=["trend_1d", "intraday_1h", "hybrid_1h", "hybrid_ml_1h"], default="hybrid_1h")
    watch.add_argument("--interval")
    watch.add_argument("--quote-order-qty", type=float, required=True)
    watch.add_argument("--poll-seconds", type=int, default=60)
    watch.add_argument("--test-order", action="store_true")
    watch.add_argument("--max-iterations", type=int)

    paper_watch = subparsers.add_parser("paper-watch", help="Sonde le marche en boucle et execute en portefeuille papier local.")
    paper_watch.add_argument("--symbol", required=True)
    paper_watch.add_argument("--preset", choices=["trend_1d", "intraday_1h", "hybrid_1h", "hybrid_ml_1h"], default="hybrid_1h")
    paper_watch.add_argument("--interval")
    paper_watch.add_argument("--quote-order-qty", type=float, required=True)
    paper_watch.add_argument("--poll-seconds", type=int, default=60)
    paper_watch.add_argument("--max-iterations", type=int)
    paper_watch.add_argument("--portfolio-name", default="default")
    paper_watch.add_argument("--initial-cash", type=float, default=10000.0)

    return parser.parse_args()


def parse_date_to_ms(value: str, *, end_of_day: bool = False) -> int:
    date = datetime.strptime(value, "%Y-%m-%d")
    if end_of_day:
        date = date.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(date.replace(tzinfo=timezone.utc).timestamp() * 1000)


def resolve_interval(args: argparse.Namespace, default_interval: str) -> str:
    return args.interval or default_interval


def resolve_risk_parameters(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.preset == "intraday_1h":
        return (
            args.fee_rate if args.fee_rate is not None else 0.001,
            args.stop_loss if args.stop_loss is not None else 0.015,
            args.take_profit if args.take_profit is not None else 0.035,
            args.position_fraction if args.position_fraction is not None else 0.80,
        )
    if args.preset == "hybrid_1h":
        return (
            args.fee_rate if args.fee_rate is not None else 0.001,
            args.stop_loss if args.stop_loss is not None else 0.018,
            args.take_profit if args.take_profit is not None else 0.050,
            args.position_fraction if args.position_fraction is not None else 0.85,
        )
    if args.preset == "hybrid_ml_1h":
        return (
            args.fee_rate if args.fee_rate is not None else 0.001,
            args.stop_loss if args.stop_loss is not None else 0.018,
            args.take_profit if args.take_profit is not None else 0.050,
            args.position_fraction if args.position_fraction is not None else 0.85,
        )
    return (
        args.fee_rate if args.fee_rate is not None else 0.001,
        args.stop_loss if args.stop_loss is not None else 0.04,
        args.take_profit if args.take_profit is not None else 0.10,
        args.position_fraction if args.position_fraction is not None else 0.95,
    )


def compute_warmup_start_ms(start_ms: int, interval: str, warmup_period: int) -> int:
    if interval.endswith("d"):
        unit = int(interval[:-1] or "1")
        delta = timedelta(days=unit * (warmup_period + 10))
        return start_ms - int(delta.total_seconds() * 1000)
    if interval.endswith("h"):
        unit = int(interval[:-1] or "1")
        delta = timedelta(hours=unit * (warmup_period + 50))
        return start_ms - int(delta.total_seconds() * 1000)
    return start_ms


def run_backtest(args: argparse.Namespace) -> None:
    client = BinanceClient()
    strategy_config = build_strategy_config(args.preset)
    strategy = TrendFollowingStrategy(strategy_config)
    fee_rate, stop_loss, take_profit, position_fraction = resolve_risk_parameters(args)
    risk = build_risk_config(
        args.preset,
        fee_rate=fee_rate,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_fraction=position_fraction,
    )
    backtester = Backtester(strategy=strategy, risk=risk)
    interval = resolve_interval(args, strategy_config.default_interval)

    start_ms = parse_date_to_ms(args.start)
    end_ms = parse_date_to_ms(args.end, end_of_day=True)
    fetch_start_ms = compute_warmup_start_ms(start_ms, interval, strategy.runtime_lookback_bars())

    results = []
    for symbol in args.symbols:
        candles = client.get_klines(symbol=symbol, interval=interval, start_time=fetch_start_ms, end_time=end_ms)
        if len(candles) <= strategy.warmup_period():
            raise RuntimeError(f"Pas assez de bougies pour backtester {symbol}.")
        result = backtester.run(
            symbol=symbol,
            interval=interval,
            candles=candles,
            initial_capital=args.capital,
            report_start_time=start_ms,
        )
        results.append(result)

    for result in results:
        print("=" * 72)
        print(f"{result.symbol} [{result.interval}] {result.start} -> {result.end}")
        print(f"Capital initial : {result.initial_capital:.2f} USDT")
        print(f"Capital final   : {result.final_equity:.2f} USDT")
        print(f"Gain total      : {result.total_gain:.2f} USDT")
        print(f"Performance     : {result.total_return_pct:.2f}%")
        print(f"Hold pur final  : {result.pure_hold_final_equity:.2f} USDT")
        print(f"Hold pur gain   : {result.pure_hold_final_equity - result.initial_capital:.2f} USDT")
        print(f"Hold pur perf   : {result.buy_and_hold_return_pct:.2f}%")
        print(f"Max drawdown    : {result.max_drawdown_pct:.2f}%")
        print(f"Trades          : {len(result.trades)}")
        if result.trades:
            last_trade = result.trades[-1]
            print(
                "Dernier trade   : "
                f"{last_trade.side} @ {last_trade.price:.2f} "
                f"({datetime.fromtimestamp(last_trade.timestamp / 1000, tz=timezone.utc).date().isoformat()})"
            )

    combined_initial = sum(result.initial_capital for result in results)
    combined_final = sum(result.final_equity for result in results)
    combined_hold_final = sum(result.pure_hold_final_equity for result in results)
    print("=" * 72)
    print("Portefeuille combine")
    print(f"Capital initial : {combined_initial:.2f} USDT")
    print(f"Capital final   : {combined_final:.2f} USDT")
    print(f"Gain total      : {combined_final - combined_initial:.2f} USDT")
    print(f"Performance     : {((combined_final / combined_initial) - 1) * 100:.2f}%")
    print(f"Hold pur final  : {combined_hold_final:.2f} USDT")
    print(f"Hold pur gain   : {combined_hold_final - combined_initial:.2f} USDT")
    print(f"Hold pur perf   : {((combined_hold_final / combined_initial) - 1) * 100:.2f}%")


def run_signal(args: argparse.Namespace) -> None:
    client = BinanceClient()
    strategy_config = build_strategy_config(args.preset)
    strategy = TrendFollowingStrategy(strategy_config)
    interval = resolve_interval(args, strategy_config.default_interval)
    evaluation = evaluate_latest_signal(client, strategy, args.symbol, interval)
    signal = evaluation["signal"]

    print(f"Symbole         : {args.symbol}")
    print(f"Preset          : {args.preset}")
    print(f"Intervalle      : {interval}")
    print(f"Action          : {signal.action}")
    print(f"Raison          : {signal.reason}")
    print(f"Score haussier  : {signal.bullish_score}")
    print(f"Score baissier  : {signal.bearish_score}")
    print("Indicateurs     :")
    for name, value in signal.indicators.items():
        print(f"  - {name}: {value:.6f}")


def run_live(args: argparse.Namespace) -> None:
    client = BinanceClient()
    if not client.api_key or not client.api_secret:
        raise RuntimeError("Le mode live requiert BINANCE_API_KEY et BINANCE_API_SECRET, meme avec --test-order.")
    strategy_config = build_strategy_config(args.preset)
    strategy = TrendFollowingStrategy(strategy_config)
    interval = resolve_interval(args, strategy_config.default_interval)
    result = maybe_execute_live_order(
        client=client,
        strategy=strategy,
        symbol=args.symbol,
        interval=interval,
        quote_order_qty=args.quote_order_qty,
        test_order=args.test_order,
    )
    signal = result["signal"]

    print(f"Symbole         : {args.symbol}")
    print(f"Preset          : {args.preset}")
    print(f"Action signal   : {signal.action}")
    print(f"Raison          : {signal.reason}")
    print(f"Execution       : {result['executed']}")
    print(f"Message         : {result['reason']}")
    if result.get("exchange_response") is not None:
        print(f"Reponse Binance : {result['exchange_response']}")


def run_watch(args: argparse.Namespace) -> None:
    client = BinanceClient()
    if not client.api_key or not client.api_secret:
        raise RuntimeError("Le mode watch requiert BINANCE_API_KEY et BINANCE_API_SECRET, meme avec --test-order.")
    strategy_config = build_strategy_config(args.preset)
    strategy = TrendFollowingStrategy(strategy_config)
    interval = resolve_interval(args, strategy_config.default_interval)
    watch_market(
        client=client,
        strategy=strategy,
        symbol=args.symbol,
        interval=interval,
        quote_order_qty=args.quote_order_qty,
        poll_seconds=args.poll_seconds,
        test_order=args.test_order,
        preset=args.preset,
        max_iterations=args.max_iterations,
    )


def run_paper_watch(args: argparse.Namespace) -> None:
    client = BinanceClient()
    strategy_config = build_strategy_config(args.preset)
    strategy = TrendFollowingStrategy(strategy_config)
    interval = resolve_interval(args, strategy_config.default_interval)
    paper = PaperBroker(
        portfolio_name=args.portfolio_name,
        initial_cash_usdt=args.initial_cash,
    )
    state_store = RuntimeStateStore()
    state_key = f"paper_{args.symbol}_{interval}_{args.preset}_{args.portfolio_name}"
    iteration = 0

    while True:
        iteration += 1
        in_position = paper.in_position(args.symbol)
        evaluation = evaluate_latest_signal(
            client=client,
            strategy=strategy,
            symbol=args.symbol,
            interval=interval,
            force_in_position=in_position,
        )
        signal = evaluation["signal"]
        latest_candle = evaluation["latest_candle"]
        state = state_store.load(state_key)
        last_processed_candle = state.get("last_processed_candle")

        if latest_candle.open_time == last_processed_candle:
            portfolio = paper.load_portfolio()
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{args.symbol} {interval} candle={latest_candle.open_datetime.isoformat()} "
                f"signal={signal.action} regime={signal.regime} executed=False "
                f"cash={portfolio['cash_usdt']:.2f} reason=Cette bougie a deja ete traitee."
            )
        else:
            result = paper.execute_signal(
                symbol=args.symbol,
                signal=signal,
                market_price=latest_candle.close,
                quote_order_qty=args.quote_order_qty,
                candle_open_time=latest_candle.open_time,
            )
            portfolio = result["portfolio"]
            state_store.save(
                state_key,
                {
                    "last_processed_candle": latest_candle.open_time,
                    "last_signal": signal.action,
                    "last_reason": result["reason"],
                },
            )
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{args.symbol} {interval} candle={latest_candle.open_datetime.isoformat()} "
                f"signal={signal.action} regime={signal.regime} executed={result['executed']} "
                f"cash={portfolio['cash_usdt']:.2f} position_open={paper.in_position(args.symbol)} "
                f"reason={result['reason']}"
            )

            paper.append_event(
                {
                    "timestamp": int(time.time()),
                    "candle_open_time": latest_candle.open_time,
                    "symbol": args.symbol,
                    "interval": interval,
                    "signal": signal.action,
                    "regime": signal.regime,
                    "executed": result["executed"],
                    "reason": result["reason"],
                    "market_price": latest_candle.close,
                    "cash_usdt": portfolio["cash_usdt"],
                    "position_open": paper.in_position(args.symbol),
                }
            )

        if args.max_iterations is not None and iteration >= args.max_iterations:
            return
        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    if args.command == "backtest":
        run_backtest(args)
        return
    if args.command == "signal":
        run_signal(args)
        return
    if args.command == "live":
        run_live(args)
        return
    if args.command == "watch":
        run_watch(args)
        return
    if args.command == "paper-watch":
        run_paper_watch(args)
        return
    raise RuntimeError(f"Commande inconnue: {args.command}")


if __name__ == "__main__":
    main()
