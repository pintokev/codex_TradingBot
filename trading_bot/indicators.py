from __future__ import annotations


def sma(values: list[float], period: int) -> list[float]:
    result: list[float] = []
    window: list[float] = []
    rolling_sum = 0.0
    for value in values:
        window.append(value)
        rolling_sum += value
        if len(window) > period:
            rolling_sum -= window.pop(0)
        result.append(rolling_sum / len(window))
    return result


def ema(values: list[float], period: int) -> list[float]:
    result: list[float] = []
    multiplier = 2.0 / (period + 1)
    current = None
    for value in values:
        current = value if current is None else (value * multiplier) + (current * (1 - multiplier))
        result.append(current)
    return result


def rsi(values: list[float], period: int = 14) -> list[float]:
    if not values:
        return []
    gains: list[float] = []
    losses: list[float] = []
    result = [50.0]
    avg_gain = None
    avg_loss = None

    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        gains.append(gain)
        losses.append(loss)

        if index < period:
            result.append(50.0)
            continue

        if index == period:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
        else:
            assert avg_gain is not None
            assert avg_loss is not None
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0:
            result.append(100.0)
            continue

        relative_strength = avg_gain / avg_loss
        result.append(100.0 - (100.0 / (1.0 + relative_strength)))

    return result


def macd(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    ema_12 = ema(values, 12)
    ema_26 = ema(values, 26)
    macd_line = [fast - slow for fast, slow in zip(ema_12, ema_26)]
    signal_line = ema(macd_line, 9)
    histogram = [line - signal for line, signal in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    if not highs:
        return []

    true_ranges: list[float] = []
    for index in range(len(highs)):
        if index == 0:
            true_range = highs[index] - lows[index]
        else:
            true_range = max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        true_ranges.append(true_range)

    return ema(true_ranges, period)
