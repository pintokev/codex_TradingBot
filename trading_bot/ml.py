from __future__ import annotations

import math


class LogisticGate:
    def __init__(self, weights: list[float], bias: float, means: list[float], stds: list[float]) -> None:
        self.weights = weights
        self.bias = bias
        self.means = means
        self.stds = stds

    @classmethod
    def fit(
        cls,
        features: list[list[float]],
        labels: list[int],
        *,
        epochs: int = 250,
        learning_rate: float = 0.08,
        l2: float = 0.0005,
    ) -> "LogisticGate":
        feature_count = len(features[0])
        means = []
        stds = []
        normalized_features: list[list[float]] = []

        for column in range(feature_count):
            values = [row[column] for row in features]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance) or 1.0
            means.append(mean)
            stds.append(std)

        for row in features:
            normalized_features.append(
                [(value - means[index]) / stds[index] for index, value in enumerate(row)]
            )

        weights = [0.0] * feature_count
        bias = 0.0

        for _ in range(epochs):
            grad_w = [0.0] * feature_count
            grad_b = 0.0
            sample_count = len(normalized_features)

            for row, label in zip(normalized_features, labels):
                score = sum(weight * value for weight, value in zip(weights, row)) + bias
                prediction = _sigmoid(score)
                error = prediction - label
                for index, value in enumerate(row):
                    grad_w[index] += error * value
                grad_b += error

            for index in range(feature_count):
                grad_w[index] = (grad_w[index] / sample_count) + (l2 * weights[index])
                weights[index] -= learning_rate * grad_w[index]
            bias -= learning_rate * (grad_b / sample_count)

        return cls(weights=weights, bias=bias, means=means, stds=stds)

    def predict_proba(self, row: list[float]) -> float:
        normalized = [
            (value - self.means[index]) / self.stds[index]
            for index, value in enumerate(row)
        ]
        score = sum(weight * value for weight, value in zip(self.weights, normalized)) + self.bias
        return _sigmoid(score)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
