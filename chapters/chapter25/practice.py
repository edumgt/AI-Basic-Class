"""크로스 엔트로피 손실 계산"""
from __future__ import annotations

import numpy as np


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    n = len(y_true)
    chosen = probs[np.arange(n), y_true]
    return float(-np.mean(np.log(chosen + 1e-12)))


def run() -> dict:
    probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.05, 0.90, 0.05],
            [0.15, 0.35, 0.50],
            [0.60, 0.25, 0.15],
        ]
    )
    y_true = np.array([0, 1, 2, 1])
    loss = cross_entropy(probs, y_true)

    return {
        "chapter": "chapter25",
        "topic": "크로스 엔트로피",
        "probs": probs.tolist(),
        "labels": y_true.tolist(),
        "loss": round(loss, 6),
    }


if __name__ == "__main__":
    print(run())
