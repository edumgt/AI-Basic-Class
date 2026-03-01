# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""초기화 스케일과 L2 정규화 영향"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `run_once` 함수를 만들어요.
def run_once(scale: float, l2: float) -> tuple[float, float]:
    # 설명: 다음 코드를 실행해요.
    np.random.seed(0)
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array([[1.0, 0.2], [0.3, 1.2], [1.3, 0.7], [0.4, 0.4]])
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([1, 0, 1, 0])

    # 설명: 값을 저장하거나 바꿔요.
    W = np.random.randn(2, 2) * scale
    # 설명: 값을 저장하거나 바꿔요.
    b = np.zeros((1, 2))

    # 설명: `softmax` 함수를 만들어요.
    def softmax(logits: np.ndarray) -> np.ndarray:
        # 설명: 값을 저장하거나 바꿔요.
        s = logits - np.max(logits, axis=1, keepdims=True)
        # 설명: 값을 저장하거나 바꿔요.
        e = np.exp(s)
        # 설명: 함수 결과를 돌려줘요.
        return e / np.sum(e, axis=1, keepdims=True)

    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(120):
        # 설명: 값을 저장하거나 바꿔요.
        logits = X @ W + b
        # 설명: 값을 저장하거나 바꿔요.
        probs = softmax(logits)
        # 설명: 값을 저장하거나 바꿔요.
        y_oh = np.eye(2)[y]

        # 설명: 값을 저장하거나 바꿔요.
        n = len(y)
        # 설명: 값을 저장하거나 바꿔요.
        dlogits = (probs - y_oh) / n
        # 설명: 값을 저장하거나 바꿔요.
        dW = X.T @ dlogits + l2 * W
        # 설명: 값을 저장하거나 바꿔요.
        db = np.sum(dlogits, axis=0, keepdims=True)

        # 설명: 값을 저장하거나 바꿔요.
        W -= 0.3 * dW
        # 설명: 값을 저장하거나 바꿔요.
        b -= 0.3 * db

    # 설명: 값을 저장하거나 바꿔요.
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    # 설명: 값을 저장하거나 바꿔요.
    reg = 0.5 * l2 * float(np.sum(W * W))
    # 설명: 함수 결과를 돌려줘요.
    return float(ce + reg), float(np.linalg.norm(W))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    weak_init = run_once(scale=0.01, l2=0.0)
    # 설명: 값을 저장하거나 바꿔요.
    strong_init = run_once(scale=1.0, l2=0.0)
    # 설명: 값을 저장하거나 바꿔요.
    with_l2 = run_once(scale=1.0, l2=0.1)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter29",
        # 설명: 다음 코드를 실행해요.
        "topic": "초기화/정규화",
        # 설명: 다음 코드를 실행해요.
        "weak_init_loss": round(weak_init[0], 6),
        # 설명: 다음 코드를 실행해요.
        "strong_init_loss": round(strong_init[0], 6),
        # 설명: 다음 코드를 실행해요.
        "strong_init_weight_norm": round(strong_init[1], 6),
        # 설명: 다음 코드를 실행해요.
        "l2_weight_norm": round(with_l2[1], 6),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
