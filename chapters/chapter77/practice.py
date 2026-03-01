"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "순전파와 역전파를 반복하면 작은 신경망도 점점 오차를 줄일 수 있다."
PRACTICE_30MIN = "numpy로 2층 신경망을 학습해 XOR 예제를 맞춘다."


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)))


def run() -> dict:
    # XOR 데이터
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)

    rng = np.random.default_rng(42)
    hidden_dim = 3

    W1 = rng.normal(0, 0.5, size=(2, hidden_dim))
    b1 = np.zeros((1, hidden_dim), dtype=float)
    W2 = rng.normal(0, 0.5, size=(hidden_dim, 1))
    b2 = np.zeros((1, 1), dtype=float)

    lr = 0.8
    epochs = 800

    # 초기 손실
    a1_init = _sigmoid(X @ W1 + b1)
    y_hat_init = _sigmoid(a1_init @ W2 + b2)
    loss_before = _bce_loss(y, y_hat_init)

    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = _sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_hat = _sigmoid(z2)

        # BCE + sigmoid gradient
        n = len(X)
        dz2 = (y_hat - y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * a1 * (1.0 - a1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    a1_final = _sigmoid(X @ W1 + b1)
    y_hat_final = _sigmoid(a1_final @ W2 + b2)
    loss_after = _bce_loss(y, y_hat_final)

    pred = (y_hat_final >= 0.5).astype(int)
    acc = float((pred == y).mean())

    return {
        "chapter": "chapter77",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "predictions": pred.flatten().tolist(),
        "target": y.astype(int).flatten().tolist(),
        "accuracy": round(acc, 4),
    }


if __name__ == "__main__":
    print(run())
