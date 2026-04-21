from __future__ import annotations

from datetime import datetime, timezone

from trading_bot.models import BacktestResult, RiskConfig, Trade
from trading_bot.strategy import TrendFollowingStrategy


class Backtester:
    def __init__(self, strategy: TrendFollowingStrategy, risk: RiskConfig | None = None) -> None:
        self.strategy = strategy
        self.risk = risk or RiskConfig()

    def run(
        self,
        symbol: str,
        interval: str,
        candles,
        initial_capital: float,
        report_start_time: int | None = None,
    ) -> BacktestResult:
        indicator_frame = self.strategy.build_indicator_frame(candles)
        cash = initial_capital
        quantity = 0.0
        entry_price = None
        active_stop_loss_pct = self.risk.stop_loss_pct
        active_take_profit_pct = self.risk.take_profit_pct
        trades: list[Trade] = []
        equity_curve: list[float] = []
        report_start_index = self.strategy.warmup_period()

        if report_start_time is not None:
            for index, candle in enumerate(candles):
                if candle.open_time >= report_start_time:
                    report_start_index = max(report_start_index, index)
                    break

        self.strategy.prepare_runtime(
            candles=candles,
            indicator_frame=indicator_frame,
            training_end_index=report_start_index - 1,
        )

        for index in range(self.strategy.warmup_period(), len(candles)):
            candle = candles[index]
            signal = self.strategy.generate_signal(
                candles=candles,
                indicator_frame=indicator_frame,
                index=index,
                in_position=quantity > 0,
            )

            if signal.action == "BUY" and quantity == 0:
                capital_to_deploy = cash * self.risk.position_fraction
                fee_paid = capital_to_deploy * self.risk.fee_rate
                quantity = (capital_to_deploy - fee_paid) / candle.close
                cash -= capital_to_deploy
                entry_price = candle.close
                active_stop_loss_pct = signal.stop_loss_pct if signal.stop_loss_pct is not None else self.risk.stop_loss_pct
                active_take_profit_pct = signal.take_profit_pct if signal.take_profit_pct is not None else self.risk.take_profit_pct
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="BUY",
                        timestamp=candle.open_time,
                        price=candle.close,
                        quantity=quantity,
                        notional=capital_to_deploy,
                        fee_paid=fee_paid,
                        reason=signal.reason,
                    )
                )

            elif quantity > 0 and signal.stop_loss_pct is not None:
                active_stop_loss_pct = signal.stop_loss_pct

            if quantity > 0 and signal.take_profit_pct is not None:
                active_take_profit_pct = signal.take_profit_pct

            if quantity > 0 and entry_price is not None and candle.close <= entry_price * (1 - active_stop_loss_pct):
                gross_notional = quantity * candle.close
                fee_paid = gross_notional * self.risk.fee_rate
                cash += gross_notional - fee_paid
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="SELL",
                        timestamp=candle.open_time,
                        price=candle.close,
                        quantity=quantity,
                        notional=gross_notional,
                        fee_paid=fee_paid,
                        reason="Stop-loss declenche.",
                    )
                )
                quantity = 0.0
                entry_price = None

            elif quantity > 0 and entry_price is not None and candle.close >= entry_price * (1 + active_take_profit_pct):
                gross_notional = quantity * candle.close
                fee_paid = gross_notional * self.risk.fee_rate
                cash += gross_notional - fee_paid
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="SELL",
                        timestamp=candle.open_time,
                        price=candle.close,
                        quantity=quantity,
                        notional=gross_notional,
                        fee_paid=fee_paid,
                        reason="Take-profit declenche.",
                    )
                )
                quantity = 0.0
                entry_price = None

            elif signal.action == "SELL" and quantity > 0:
                gross_notional = quantity * candle.close
                fee_paid = gross_notional * self.risk.fee_rate
                cash += gross_notional - fee_paid
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="SELL",
                        timestamp=candle.open_time,
                        price=candle.close,
                        quantity=quantity,
                        notional=gross_notional,
                        fee_paid=fee_paid,
                        reason=signal.reason,
                    )
                )
                quantity = 0.0
                entry_price = None

            if index >= report_start_index:
                equity_curve.append(cash + (quantity * candle.close))

        final_close = candles[-1].close
        final_equity = cash + (quantity * final_close * (1 - self.risk.fee_rate))
        pure_hold_final_equity = initial_capital * (candles[-1].close / candles[report_start_index].close)
        buy_and_hold_return_pct = ((pure_hold_final_equity / initial_capital) - 1) * 100

        start = self._format_timestamp(candles[report_start_index].open_time)
        end = self._format_timestamp(candles[-1].open_time)
        return BacktestResult(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            initial_capital=initial_capital,
            final_equity=final_equity,
            pure_hold_final_equity=pure_hold_final_equity,
            total_return_pct=((final_equity / initial_capital) - 1) * 100,
            buy_and_hold_return_pct=buy_and_hold_return_pct,
            max_drawdown_pct=self._compute_max_drawdown(equity_curve),
            trades=trades,
        )

    @staticmethod
    def _format_timestamp(timestamp_ms: int) -> str:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        peak = None
        max_drawdown = 0.0
        for equity in equity_curve:
            if peak is None or equity > peak:
                peak = equity
            if peak and peak > 0:
                drawdown = ((peak - equity) / peak) * 100
                max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown
