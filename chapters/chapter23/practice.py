"""활성화 함수 비교"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run() -> dict:
    x = np.linspace(-3, 3, 7)
    relu = np.maximum(0, x)
    sig = sigmoid(x)
    tanh = np.tanh(x)

    return {
        "chapter": "chapter23",
        "topic": "활성화 함수 비교",
        "x": np.round(x, 3).tolist(),
        "relu": np.round(relu, 3).tolist(),
        "sigmoid": np.round(sig, 3).tolist(),
        "tanh": np.round(tanh, 3).tolist(),
    }


if __name__ == "__main__":
    print(run())
