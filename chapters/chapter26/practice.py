# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""단층 분류기 역전파 기초"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `softmax` 함수를 만들어요.
def softmax(logits: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    # 설명: 값을 저장하거나 바꿔요.
    ex = np.exp(shifted)
    # 설명: 함수 결과를 돌려줘요.
    return ex / np.sum(ex, axis=1, keepdims=True)


# 설명: `one_hot` 함수를 만들어요.
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    out = np.zeros((len(y), num_classes))
    # 설명: 값을 저장하거나 바꿔요.
    out[np.arange(len(y)), y] = 1.0
    # 설명: 함수 결과를 돌려줘요.
    return out


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 0.3]])
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([0, 1, 0])

    # 설명: 값을 저장하거나 바꿔요.
    W = np.array([[0.2, -0.1], [0.1, 0.3]])
    # 설명: 값을 저장하거나 바꿔요.
    b = np.zeros((1, 2))

    # 설명: 값을 저장하거나 바꿔요.
    logits = X @ W + b
    # 설명: 값을 저장하거나 바꿔요.
    probs = softmax(logits)
    # 설명: 값을 저장하거나 바꿔요.
    y_oh = one_hot(y, 2)

    # 설명: 값을 저장하거나 바꿔요.
    n = X.shape[0]
    # 설명: 값을 저장하거나 바꿔요.
    dlogits = (probs - y_oh) / n
    # 설명: 값을 저장하거나 바꿔요.
    dW = X.T @ dlogits
    # 설명: 값을 저장하거나 바꿔요.
    db = np.sum(dlogits, axis=0, keepdims=True)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter26",
        # 설명: 다음 코드를 실행해요.
        "topic": "역전파 기울기",
        # 설명: 다음 코드를 실행해요.
        "probs": np.round(probs, 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "dW": np.round(dW, 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "db": np.round(db, 4).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
