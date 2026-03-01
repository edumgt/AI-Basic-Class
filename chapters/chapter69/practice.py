# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""소프트맥스 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "소프트맥스는 점수(logit)를 확률로 바꿔 총합이 1이 되게 만든다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "벡터를 softmax로 변환하고 합이 1인지 확인한다."


# 설명: `softmax` 함수를 만들어요.
def softmax(logits: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    # 설명: 값을 저장하거나 바꿔요.
    exp_values = np.exp(shifted)
    # 설명: 함수 결과를 돌려줘요.
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    logits = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [2.2, 0.7, -1.0],
            # 설명: 다음 코드를 실행해요.
            [0.1, 1.9, 1.2],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    probs = softmax(logits)
    # 설명: 값을 저장하거나 바꿔요.
    shifted_probs = softmax(logits + 10.0)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter69",
        # 설명: 다음 코드를 실행해요.
        "topic": "소프트맥스",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "logits": logits.tolist(),
        # 설명: 다음 코드를 실행해요.
        "probabilities": probs.round(4).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "row_sum": probs.sum(axis=1).round(4).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "argmax_class": np.argmax(probs, axis=1).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "shift_invariant_check": bool(np.allclose(probs, shifted_probs, atol=1e-9)),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
