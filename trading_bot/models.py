from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    num_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @property
    def open_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.open_time / 1000, tz=timezone.utc)


@dataclass
class StrategyConfig:
    name: str = "trend_1d"
    fast_ema_period: int = 20
    slow_ema_period: int = 50
    trend_sma_period: int = 200
    rsi_period: int = 14
    breakout_lookback: int = 20
    bullish_score_threshold: int = 3
    bearish_score_threshold: int = 2
    rsi_entry_min: float = 52.0
    rsi_entry_max: float = 72.0
    rsi_exit_max: float = 48.0
    volume_sma_period: int = 20
    min_volume_ratio: float = 0.0
    breakout_buffer: float = 0.995
    default_interval: str = "1d"
    ml_training_window: int = 0
    ml_horizon: int = 0
    ml_return_target: float = 0.0
    ml_min_probability: float = 0.0


@dataclass
class RiskConfig:
    fee_rate: float = 0.001
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.18
    position_fraction: float = 0.95


@dataclass
class Trade:
    symbol: str
    side: str
    timestamp: int
    price: float
    quantity: float
    notional: float
    fee_paid: float
    reason: str


@dataclass
class Signal:
    action: str
    reason: str
    bullish_score: int
    bearish_score: int
    indicators: dict[str, float]
    regime: str = "unknown"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class BacktestResult:
    symbol: str
    interval: str
    start: str
    end: str
    initial_capital: float
    final_equity: float
    pure_hold_final_equity: float
    total_return_pct: float
    buy_and_hold_return_pct: float
    max_drawdown_pct: float
    trades: list[Trade] = field(default_factory=list)

    @property
    def total_gain(self) -> float:
        return self.final_equity - self.initial_capital
