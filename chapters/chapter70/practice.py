"""손실함수 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "손실함수는 예측이 얼마나 틀렸는지 숫자로 알려준다."
PRACTICE_30MIN = "MSE와 Cross-Entropy를 직접 계산한다."


def run() -> dict:
    # 회귀 예시
    y_true_reg = np.array([3.0, 5.0, 2.5, 7.0], dtype=float)
    y_pred_reg = np.array([2.8, 4.6, 3.1, 6.2], dtype=float)
    mse = float(np.mean((y_true_reg - y_pred_reg) ** 2))

    # 분류 예시(3클래스)
    y_true_cls = np.array([0, 2, 1], dtype=int)
    y_prob = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.20, 0.70],
            [0.25, 0.60, 0.15],
        ],
        dtype=float,
    )
    eps = 1e-9
    chosen_prob = y_prob[np.arange(len(y_true_cls)), y_true_cls]
    cross_entropy = float(-np.mean(np.log(chosen_prob + eps)))

    return {
        "chapter": "chapter70",
        "topic": "손실함수",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "mse": round(mse, 6),
        "cross_entropy": round(cross_entropy, 6),
        "chosen_probabilities": chosen_prob.round(4).tolist(),
    }


if __name__ == "__main__":
    print(run())
