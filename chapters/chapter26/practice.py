"""단층 분류기 역전파 기초"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def run() -> dict:
    X = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 0.3]])
    y = np.array([0, 1, 0])

    W = np.array([[0.2, -0.1], [0.1, 0.3]])
    b = np.zeros((1, 2))

    logits = X @ W + b
    probs = softmax(logits)
    y_oh = one_hot(y, 2)

    n = X.shape[0]
    dlogits = (probs - y_oh) / n
    dW = X.T @ dlogits
    db = np.sum(dlogits, axis=0, keepdims=True)

    return {
        "chapter": "chapter26",
        "topic": "역전파 기울기",
        "probs": np.round(probs, 4).tolist(),
        "dW": np.round(dW, 4).tolist(),
        "db": np.round(db, 4).tolist(),
    }


if __name__ == "__main__":
    print(run())
