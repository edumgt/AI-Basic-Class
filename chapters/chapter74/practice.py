"""초기화와 학습 안정성 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "초기 파라미터에 따라 같은 모델도 학습 속도와 안정성이 달라질 수 있다."
PRACTICE_30MIN = "서로 다른 초기값으로 선형회귀를 학습해 손실 추세를 비교한다."


def _train_linear(init_w: float, init_b: float, lr: float = 0.01, epochs: int = 200) -> dict:
    x = np.arange(1, 9, dtype=float)
    y = 2.0 * x + 1.0

    w = init_w
    b = init_b
    losses = []

    n = len(x)
    for epoch in range(epochs):
        pred = w * x + b
        err = pred - y
        loss = float(np.mean(err**2))

        dw = float((2.0 / n) * np.sum(err * x))
        db = float((2.0 / n) * np.sum(err))

        w -= lr * dw
        b -= lr * db

        if epoch < 5 or epoch >= epochs - 5:
            losses.append(round(loss, 6))

    final_loss = float(np.mean((w * x + b - y) ** 2))

    return {
        "init": {"w": init_w, "b": init_b},
        "final": {"w": round(float(w), 5), "b": round(float(b), 5)},
        "final_loss": round(final_loss, 8),
        "sampled_losses": losses,
    }


def run() -> dict:
    near_init = _train_linear(init_w=1.8, init_b=0.6)
    far_init = _train_linear(init_w=-8.0, init_b=10.0)

    return {
        "chapter": "chapter74",
        "topic": "초기화와 학습 안정성",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "near_init_result": near_init,
        "far_init_result": far_init,
    }


if __name__ == "__main__":
    print(run())
