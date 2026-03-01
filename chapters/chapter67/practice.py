"""인공신경망이란 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "뉴런은 입력에 가중치를 곱해 더한 뒤 출력값을 만든다."
PRACTICE_30MIN = "numpy로 1층 순전파를 계산하고 출력 형태를 확인한다."


def run() -> dict:
    # 3개 샘플, 각 샘플은 4개 특성
    X = np.array(
        [
            [0.2, 0.7, 1.0, -0.3],
            [1.2, -0.4, 0.3, 0.8],
            [0.0, 0.5, -0.6, 1.1],
        ],
        dtype=float,
    )

    # 1층(입력 4 -> 출력 3)
    W = np.array(
        [
            [0.4, -0.2, 0.1],
            [0.7, 0.3, -0.5],
            [-0.6, 0.8, 0.2],
            [0.2, -0.1, 0.9],
        ],
        dtype=float,
    )
    b = np.array([0.1, -0.2, 0.05], dtype=float)

    logits = X @ W + b
    relu_out = np.maximum(0.0, logits)

    return {
        "chapter": "chapter67",
        "topic": "인공신경망이란",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "input_shape": list(X.shape),
        "weight_shape": list(W.shape),
        "output_shape": list(logits.shape),
        "logits": logits.round(4).tolist(),
        "relu_output": relu_out.round(4).tolist(),
    }


if __name__ == "__main__":
    print(run())
