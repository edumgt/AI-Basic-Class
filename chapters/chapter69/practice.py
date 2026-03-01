"""소프트맥스 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "소프트맥스는 점수(logit)를 확률로 바꿔 총합이 1이 되게 만든다."
PRACTICE_30MIN = "벡터를 softmax로 변환하고 합이 1인지 확인한다."


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def run() -> dict:
    logits = np.array(
        [
            [2.2, 0.7, -1.0],
            [0.1, 1.9, 1.2],
        ],
        dtype=float,
    )

    probs = softmax(logits)
    shifted_probs = softmax(logits + 10.0)

    return {
        "chapter": "chapter69",
        "topic": "소프트맥스",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "logits": logits.tolist(),
        "probabilities": probs.round(4).tolist(),
        "row_sum": probs.sum(axis=1).round(4).tolist(),
        "argmax_class": np.argmax(probs, axis=1).tolist(),
        "shift_invariant_check": bool(np.allclose(probs, shifted_probs, atol=1e-9)),
    }


if __name__ == "__main__":
    print(run())
