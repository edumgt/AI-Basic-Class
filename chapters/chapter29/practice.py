"""초기화 스케일과 L2 정규화 영향"""
from __future__ import annotations

import numpy as np


def run_once(scale: float, l2: float) -> tuple[float, float]:
    np.random.seed(0)
    X = np.array([[1.0, 0.2], [0.3, 1.2], [1.3, 0.7], [0.4, 0.4]])
    y = np.array([1, 0, 1, 0])

    W = np.random.randn(2, 2) * scale
    b = np.zeros((1, 2))

    def softmax(logits: np.ndarray) -> np.ndarray:
        s = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=1, keepdims=True)

    for _ in range(120):
        logits = X @ W + b
        probs = softmax(logits)
        y_oh = np.eye(2)[y]

        n = len(y)
        dlogits = (probs - y_oh) / n
        dW = X.T @ dlogits + l2 * W
        db = np.sum(dlogits, axis=0, keepdims=True)

        W -= 0.3 * dW
        b -= 0.3 * db

    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    reg = 0.5 * l2 * float(np.sum(W * W))
    return float(ce + reg), float(np.linalg.norm(W))


def run() -> dict:
    weak_init = run_once(scale=0.01, l2=0.0)
    strong_init = run_once(scale=1.0, l2=0.0)
    with_l2 = run_once(scale=1.0, l2=0.1)

    return {
        "chapter": "chapter29",
        "topic": "초기화/정규화",
        "weak_init_loss": round(weak_init[0], 6),
        "strong_init_loss": round(strong_init[0], 6),
        "strong_init_weight_norm": round(strong_init[1], 6),
        "l2_weight_norm": round(with_l2[1], 6),
    }


if __name__ == "__main__":
    print(run())
