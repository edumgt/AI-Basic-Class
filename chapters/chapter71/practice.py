"""경사하강법 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "경사하강법은 기울기를 따라 손실이 줄어드는 방향으로 조금씩 이동한다."
PRACTICE_30MIN = "2차 함수의 최소점으로 이동하는 과정을 기록한다."


def _loss(x: float) -> float:
    return (x - 3.0) ** 2 + 2.0


def _grad(x: float) -> float:
    return 2.0 * (x - 3.0)


def run() -> dict:
    lr = 0.2
    steps = 15
    x = -5.0

    trajectory = []
    for step in range(steps):
        g = _grad(x)
        x = x - lr * g
        trajectory.append(
            {
                "step": step + 1,
                "x": round(float(x), 5),
                "loss": round(float(_loss(x)), 5),
            }
        )

    return {
        "chapter": "chapter71",
        "topic": "경사하강법",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "learning_rate": lr,
        "start_x": -5.0,
        "end_x": round(float(x), 6),
        "end_loss": round(float(_loss(x)), 6),
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    print(run())
