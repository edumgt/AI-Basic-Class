"""활성화 함수 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "활성화 함수는 직선만으로는 못 배우는 복잡한 패턴을 학습하게 돕는다."
PRACTICE_30MIN = "ReLU, Sigmoid, Tanh 값을 같은 입력에서 비교한다."


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run() -> dict:
    x = np.linspace(-3, 3, 13)
    relu = np.maximum(0.0, x)
    sigmoid = _sigmoid(x)
    tanh = np.tanh(x)

    sample_idx = [2, 4, 6, 8, 10]
    table = [
        {
            "x": round(float(x[i]), 2),
            "relu": round(float(relu[i]), 4),
            "sigmoid": round(float(sigmoid[i]), 4),
            "tanh": round(float(tanh[i]), 4),
        }
        for i in sample_idx
    ]

    return {
        "chapter": "chapter68",
        "topic": "활성화 함수",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "x_values": x.round(2).tolist(),
        "relu": relu.round(4).tolist(),
        "sigmoid": sigmoid.round(4).tolist(),
        "tanh": tanh.round(4).tolist(),
        "sample_table": table,
    }


if __name__ == "__main__":
    print(run())
