"""소프트맥스 확률 해석"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def run() -> dict:
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.2, 3.1]])
    p_t1 = softmax(logits, temperature=1.0)
    p_t05 = softmax(logits, temperature=0.5)

    return {
        "chapter": "chapter24",
        "topic": "소프트맥스",
        "logits": logits.tolist(),
        "probs_t1": np.round(p_t1, 4).tolist(),
        "probs_t05": np.round(p_t05, 4).tolist(),
    }


if __name__ == "__main__":
    print(run())
