# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""소프트맥스 확률 해석"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `softmax` 함수를 만들어요.
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    scaled = logits / temperature
    # 설명: 값을 저장하거나 바꿔요.
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    # 설명: 값을 저장하거나 바꿔요.
    exp_scores = np.exp(shifted)
    # 설명: 함수 결과를 돌려줘요.
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.2, 3.1]])
    # 설명: 값을 저장하거나 바꿔요.
    p_t1 = softmax(logits, temperature=1.0)
    # 설명: 값을 저장하거나 바꿔요.
    p_t05 = softmax(logits, temperature=0.5)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter24",
        # 설명: 다음 코드를 실행해요.
        "topic": "소프트맥스",
        # 설명: 다음 코드를 실행해요.
        "logits": logits.tolist(),
        # 설명: 다음 코드를 실행해요.
        "probs_t1": np.round(p_t1, 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "probs_t05": np.round(p_t05, 4).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
