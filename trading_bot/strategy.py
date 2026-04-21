from __future__ import annotations

from trading_bot.indicators import atr, ema, macd, rsi, sma
from trading_bot.ml import LogisticGate
from trading_bot.models import Candle, RiskConfig, Signal, StrategyConfig


def build_strategy_config(preset: str) -> StrategyConfig:
    if preset == "trend_1d":
        return StrategyConfig(
            name="trend_1d",
            fast_ema_period=12,
            slow_ema_period=40,
            trend_sma_period=200,
            rsi_period=14,
            breakout_lookback=10,
            bullish_score_threshold=4,
            bearish_score_threshold=2,
            rsi_entry_min=55.0,
            rsi_entry_max=80.0,
            rsi_exit_max=47.0,
            volume_sma_period=20,
            min_volume_ratio=0.0,
            breakout_buffer=1.0,
            default_interval="1d",
        )
    if preset == "intraday_1h":
        return StrategyConfig(
            name="intraday_1h",
            fast_ema_period=12,
            slow_ema_period=34,
            trend_sma_period=200,
            rsi_period=14,
            breakout_lookback=18,
            bullish_score_threshold=3,
            bearish_score_threshold=2,
            rsi_entry_min=52.0,
            rsi_entry_max=75.0,
            rsi_exit_max=46.0,
            volume_sma_period=20,
            min_volume_ratio=1.0,
            breakout_buffer=0.999,
            default_interval="1h",
        )
    if preset == "hybrid_1h":
        return StrategyConfig(
            name="hybrid_1h",
            fast_ema_period=12,
            slow_ema_period=34,
            trend_sma_period=200,
            rsi_period=14,
            breakout_lookback=24,
            bullish_score_threshold=3,
            bearish_score_threshold=2,
            rsi_entry_min=50.0,
            rsi_entry_max=78.0,
            rsi_exit_max=45.0,
            volume_sma_period=20,
            min_volume_ratio=1.0,
            breakout_buffer=0.999,
            default_interval="1h",
        )
    if preset == "hybrid_ml_1h":
        return StrategyConfig(
            name="hybrid_ml_1h",
            fast_ema_period=12,
            slow_ema_period=34,
            trend_sma_period=200,
            rsi_period=14,
            breakout_lookback=24,
            bullish_score_threshold=3,
            bearish_score_threshold=2,
            rsi_entry_min=50.0,
            rsi_entry_max=78.0,
            rsi_exit_max=45.0,
            volume_sma_period=20,
            min_volume_ratio=1.0,
            breakout_buffer=0.999,
            default_interval="1h",
            ml_training_window=1200,
            ml_horizon=12,
            ml_return_target=0.008,
            ml_min_probability=0.45,
        )
    raise ValueError(f"Preset inconnu: {preset}")


def build_risk_config(preset: str, *, fee_rate: float, stop_loss: float, take_profit: float, position_fraction: float) -> RiskConfig:
    if preset in {"intraday_1h", "hybrid_1h", "hybrid_ml_1h"}:
        return RiskConfig(
            fee_rate=fee_rate,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            position_fraction=position_fraction,
        )
    return RiskConfig(
        fee_rate=fee_rate,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        position_fraction=position_fraction,
    )


class TrendFollowingStrategy:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self.ml_model: LogisticGate | None = None
        self.ml_probabilities: dict[int, float] = {}

    def build_indicator_frame(self, candles: list[Candle]) -> dict[str, list[float]]:
        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        volumes = [candle.volume for candle in candles]
        ema_fast = ema(closes, self.config.fast_ema_period)
        ema_slow = ema(closes, self.config.slow_ema_period)
        sma_trend = sma(closes, self.config.trend_sma_period)
        rsi_values = rsi(closes, self.config.rsi_period)
        macd_line, signal_line, histogram = macd(closes)
        volume_sma = sma(volumes, self.config.volume_sma_period)
        atr_values = atr(highs, lows, closes, period=14)
        return {
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "sma_trend": sma_trend,
            "rsi": rsi_values,
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
            "volume_sma": volume_sma,
            "atr": atr_values,
        }

    def prepare_runtime(
        self,
        candles: list[Candle],
        indicator_frame: dict[str, list[float]],
        *,
        training_end_index: int,
    ) -> None:
        self.ml_model = None
        self.ml_probabilities = {}

        if self.config.name != "hybrid_ml_1h":
            return

        horizon = self.config.ml_horizon
        if horizon <= 0:
            return

        last_train_index = training_end_index - horizon
        if last_train_index <= self.warmup_period():
            return

        first_train_index = max(self.warmup_period(), last_train_index - self.config.ml_training_window + 1)
        features: list[list[float]] = []
        labels: list[int] = []
        closes = indicator_frame["close"]

        for index in range(first_train_index, last_train_index + 1):
            feature_row = self._build_ml_features(indicator_frame, index)
            future_slice = closes[index + 1 : index + horizon + 1]
            if not future_slice:
                continue
            future_max_return = (max(future_slice) / closes[index]) - 1.0
            labels.append(int(future_max_return >= self.config.ml_return_target))
            features.append(feature_row)

        if len(features) < 80:
            return
        if len({label for label in labels}) < 2:
            return

        self.ml_model = LogisticGate.fit(features, labels)
        for index in range(self.warmup_period(), len(candles)):
            self.ml_probabilities[index] = self.ml_model.predict_proba(self._build_ml_features(indicator_frame, index))

    def generate_signal(
        self,
        candles: list[Candle],
        indicator_frame: dict[str, list[float]],
        index: int,
        in_position: bool,
    ) -> Signal:
        closes = indicator_frame["close"]
        highs = indicator_frame["high"]
        volumes = indicator_frame["volume"]
        close_price = closes[index]
        ema_fast = indicator_frame["ema_fast"][index]
        ema_slow = indicator_frame["ema_slow"][index]
        sma_trend = indicator_frame["sma_trend"][index]
        rsi_value = indicator_frame["rsi"][index]
        macd_histogram = indicator_frame["macd_histogram"][index]
        volume_sma = indicator_frame["volume_sma"][index]
        atr_value = indicator_frame["atr"][index]
        breakout_high = max(highs[index - self.config.breakout_lookback : index])
        breakout_low = min(indicator_frame["low"][index - self.config.breakout_lookback : index])
        atr_pct = (atr_value / close_price) if close_price else 0.0

        bullish_score = 0
        bearish_score = 0

        bullish_score += int(close_price > sma_trend)
        bullish_score += int(ema_fast > ema_slow)
        bullish_score += int(macd_histogram > 0)
        bullish_score += int(self.config.rsi_entry_min <= rsi_value <= self.config.rsi_entry_max)

        bearish_score += int(close_price < sma_trend)
        bearish_score += int(ema_fast < ema_slow)
        bearish_score += int(macd_histogram < 0)
        bearish_score += int(rsi_value < self.config.rsi_exit_max)

        volume_ratio = (volumes[index] / volume_sma) if volume_sma else 0.0
        indicators = {
            "index": float(index),
            "close": close_price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "sma_trend": sma_trend,
            "rsi": rsi_value,
            "macd_histogram": macd_histogram,
            "breakout_high": breakout_high,
            "breakout_low": breakout_low,
            "volume_ratio": volume_ratio,
            "atr": atr_value,
            "atr_pct": atr_pct,
        }

        if self.config.name in {"hybrid_1h", "hybrid_ml_1h"}:
            regime = self._detect_market_regime(
                close_price=close_price,
                sma_trend=sma_trend,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                macd_histogram=macd_histogram,
                rsi_value=rsi_value,
                atr_pct=atr_pct,
            )
            indicators["regime"] = {"bull": 1.0, "range": 0.0, "bear": -1.0}[regime]
            return self._generate_hybrid_signal(
                in_position=in_position,
                regime=regime,
                close_price=close_price,
                breakout_high=breakout_high,
                breakout_low=breakout_low,
                volume_ratio=volume_ratio,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                sma_trend=sma_trend,
                rsi_value=rsi_value,
                macd_histogram=macd_histogram,
                bullish_score=bullish_score,
                bearish_score=bearish_score,
                indicators=indicators,
            )

        breakout_confirmed = close_price > (breakout_high * self.config.breakout_buffer)
        volume_confirmed = volume_ratio >= self.config.min_volume_ratio

        if (
            not in_position
            and bullish_score >= self.config.bullish_score_threshold
            and breakout_confirmed
            and volume_confirmed
        ):
            return Signal(
                action="BUY",
                reason="Tendance haussiere confirmee par le score d'indicateurs.",
                bullish_score=bullish_score,
                bearish_score=bearish_score,
                indicators=indicators,
                regime="bull",
            )

        if in_position:
            if bearish_score >= self.config.bearish_score_threshold:
                return Signal(
                    action="SELL",
                    reason="Structure de tendance devenue baissiere.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime="bear",
                )

        return Signal(
            action="HOLD",
            reason="Pas de signal exploitable.",
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            indicators=indicators,
            regime="range",
        )

    def _detect_market_regime(
        self,
        *,
        close_price: float,
        sma_trend: float,
        ema_fast: float,
        ema_slow: float,
        macd_histogram: float,
        rsi_value: float,
        atr_pct: float,
    ) -> str:
        if (
            close_price > sma_trend * 1.01
            and ema_fast > ema_slow
            and macd_histogram > 0
            and rsi_value >= 54
        ):
            return "bull"
        if (
            close_price < sma_trend * 0.995
            and ema_fast < ema_slow
            and macd_histogram < 0
            and rsi_value <= 48
        ):
            return "bear"
        return "range"

    def _generate_hybrid_signal(
        self,
        *,
        in_position: bool,
        regime: str,
        close_price: float,
        breakout_high: float,
        breakout_low: float,
        volume_ratio: float,
        ema_fast: float,
        ema_slow: float,
        sma_trend: float,
        rsi_value: float,
        macd_histogram: float,
        bullish_score: int,
        bearish_score: int,
        indicators: dict[str, float],
    ) -> Signal:
        ml_probability = self.ml_probabilities.get(indicators.get("index", -1), 0.0)
        indicators["ml_probability"] = ml_probability

        bull_entry = (
            bullish_score >= 3
            and close_price > breakout_high * 0.999
            and volume_ratio >= 0.95
            and rsi_value <= 78
        )
        range_entry = (
            close_price > ema_fast
            and close_price > breakout_high * 0.997
            and rsi_value >= 50
            and rsi_value <= 65
            and macd_histogram > 0
            and volume_ratio >= 1.0
        )

        if not in_position:
            if regime == "bull" and bull_entry:
                if self.config.name == "hybrid_ml_1h" and ml_probability < max(0.35, self.config.ml_min_probability - 0.10):
                    return Signal(
                        action="HOLD",
                        reason="Filtre ML: probabilite trop faible pour confirmer le regime bull.",
                        bullish_score=bullish_score,
                        bearish_score=bearish_score,
                        indicators=indicators,
                        regime=regime,
                        stop_loss_pct=0.035,
                        take_profit_pct=0.20,
                    )
                return Signal(
                    action="BUY",
                    reason="Regime bull: achat trend-following sur cassure.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime=regime,
                    stop_loss_pct=0.035,
                    take_profit_pct=0.20,
                )
            if regime == "range" and range_entry:
                if self.config.name == "hybrid_ml_1h" and ml_probability < self.config.ml_min_probability:
                    return Signal(
                        action="HOLD",
                        reason="Filtre ML: probabilite insuffisante pour trader le range.",
                        bullish_score=bullish_score,
                        bearish_score=bearish_score,
                        indicators=indicators,
                        regime=regime,
                        stop_loss_pct=0.015,
                        take_profit_pct=0.04,
                    )
                return Signal(
                    action="BUY",
                    reason="Regime range: achat prudent sur reprise validee.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime=regime,
                    stop_loss_pct=0.015,
                    take_profit_pct=0.04,
                )

        if in_position:
            if regime == "bull" and (bearish_score >= 3 or (close_price < ema_slow and macd_histogram < 0 and rsi_value < 50)):
                return Signal(
                    action="SELL",
                    reason="Regime bull: sortie trend-following sur cassure du momentum.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime=regime,
                    stop_loss_pct=0.035,
                    take_profit_pct=0.20,
                )
            if regime == "range" and (close_price < ema_fast or rsi_value >= 66 or macd_histogram < 0):
                return Signal(
                    action="SELL",
                    reason="Regime range: prise de profit ou invalidation.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime=regime,
                    stop_loss_pct=0.015,
                    take_profit_pct=0.04,
                )
            if regime == "bear" and (close_price < ema_fast or macd_histogram < 0 or bearish_score >= 2):
                return Signal(
                    action="SELL",
                    reason="Regime bear: sortie defensive rapide.",
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    indicators=indicators,
                    regime=regime,
                    stop_loss_pct=0.012,
                    take_profit_pct=0.025,
                )

        return Signal(
            action="HOLD",
            reason="Pas de signal exploitable pour le regime courant.",
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            indicators=indicators,
            regime=regime,
            stop_loss_pct=0.012 if regime == "bear" else (0.035 if regime == "bull" else 0.015),
            take_profit_pct=0.025 if regime == "bear" else (0.20 if regime == "bull" else 0.04),
        )

    def _build_ml_features(self, indicator_frame: dict[str, list[float]], index: int) -> list[float]:
        close_price = indicator_frame["close"][index]
        sma_trend = indicator_frame["sma_trend"][index]
        ema_fast = indicator_frame["ema_fast"][index]
        ema_slow = indicator_frame["ema_slow"][index]
        rsi_value = indicator_frame["rsi"][index]
        macd_histogram = indicator_frame["macd_histogram"][index]
        volume_sma = indicator_frame["volume_sma"][index]
        volume = indicator_frame["volume"][index]
        atr_value = indicator_frame["atr"][index]
        highs = indicator_frame["high"]
        lows = indicator_frame["low"]
        breakout_high = max(highs[index - self.config.breakout_lookback : index])
        breakout_low = min(lows[index - self.config.breakout_lookback : index])
        volume_ratio = (volume / volume_sma) if volume_sma else 0.0

        return [
            (close_price / sma_trend) - 1.0 if sma_trend else 0.0,
            (ema_fast / ema_slow) - 1.0 if ema_slow else 0.0,
            rsi_value / 100.0,
            macd_histogram / close_price if close_price else 0.0,
            volume_ratio - 1.0,
            atr_value / close_price if close_price else 0.0,
            (close_price / breakout_high) - 1.0 if breakout_high else 0.0,
            (close_price / breakout_low) - 1.0 if breakout_low else 0.0,
        ]

    def runtime_lookback_bars(self) -> int:
        if self.config.name == "hybrid_ml_1h":
            return max(self.warmup_period(), self.config.ml_training_window + self.config.ml_horizon + 50)
        return self.warmup_period()

    def warmup_period(self) -> int:
        return max(
            self.config.fast_ema_period,
            self.config.slow_ema_period,
            self.config.trend_sma_period,
            self.config.breakout_lookback + 1,
        )
