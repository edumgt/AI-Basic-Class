# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""크로스 엔트로피 손실 계산"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `cross_entropy` 함수를 만들어요.
def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    # 설명: 값을 저장하거나 바꿔요.
    n = len(y_true)
    # 설명: 값을 저장하거나 바꿔요.
    chosen = probs[np.arange(n), y_true]
    # 설명: 함수 결과를 돌려줘요.
    return float(-np.mean(np.log(chosen + 1e-12)))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    probs = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [0.70, 0.20, 0.10],
            # 설명: 다음 코드를 실행해요.
            [0.05, 0.90, 0.05],
            # 설명: 다음 코드를 실행해요.
            [0.15, 0.35, 0.50],
            # 설명: 다음 코드를 실행해요.
            [0.60, 0.25, 0.15],
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([0, 1, 2, 1])
    # 설명: 값을 저장하거나 바꿔요.
    loss = cross_entropy(probs, y_true)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter25",
        # 설명: 다음 코드를 실행해요.
        "topic": "크로스 엔트로피",
        # 설명: 다음 코드를 실행해요.
        "probs": probs.tolist(),
        # 설명: 다음 코드를 실행해요.
        "labels": y_true.tolist(),
        # 설명: 다음 코드를 실행해요.
        "loss": round(loss, 6),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
