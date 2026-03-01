# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""활성화 함수 비교"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `sigmoid` 함수를 만들어요.
def sigmoid(x: np.ndarray) -> np.ndarray:
    # 설명: 함수 결과를 돌려줘요.
    return 1 / (1 + np.exp(-x))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    x = np.linspace(-3, 3, 7)
    # 설명: 값을 저장하거나 바꿔요.
    relu = np.maximum(0, x)
    # 설명: 값을 저장하거나 바꿔요.
    sig = sigmoid(x)
    # 설명: 값을 저장하거나 바꿔요.
    tanh = np.tanh(x)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter23",
        # 설명: 다음 코드를 실행해요.
        "topic": "활성화 함수 비교",
        # 설명: 다음 코드를 실행해요.
        "x": np.round(x, 3).tolist(),
        # 설명: 다음 코드를 실행해요.
        "relu": np.round(relu, 3).tolist(),
        # 설명: 다음 코드를 실행해요.
        "sigmoid": np.round(sig, 3).tolist(),
        # 설명: 다음 코드를 실행해요.
        "tanh": np.round(tanh, 3).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
