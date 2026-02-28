"""2층 신경망 fitting 루프"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    s = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def run() -> dict:
    np.random.seed(7)

    X = np.array(
        [
            [0.2, 0.1, 0.7, 0.0],
            [0.9, 0.1, 0.0, 0.3],
            [0.3, 0.8, 0.2, 0.1],
            [0.8, 0.2, 0.1, 0.4],
            [0.1, 0.7, 0.4, 0.2],
            [0.9, 0.2, 0.2, 0.8],
        ]
    )
    y = np.array([2, 0, 1, 0, 1, 0])
    y_oh = one_hot(y, 3)

    W1 = np.random.randn(4, 6) * 0.1
    b1 = np.zeros((1, 6))
    W2 = np.random.randn(6, 3) * 0.1
    b2 = np.zeros((1, 3))

    lr = 0.4
    losses = []
    for _ in range(250):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        logits = a1 @ W2 + b2
        probs = softmax(logits)

        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
        losses.append(float(loss))

        n = X.shape[0]
        dlogits = (probs - y_oh) / n
        dW2 = a1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        da1 = dlogits @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    pred = np.argmax(probs, axis=1)

    return {
        "chapter": "chapter28",
        "topic": "2층 신경망 fitting",
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "train_accuracy": round(float(np.mean(pred == y)), 4),
    }


if __name__ == "__main__":
    print(run())
