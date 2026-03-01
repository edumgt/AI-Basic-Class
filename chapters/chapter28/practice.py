# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""2층 신경망 fitting 루프"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `softmax` 함수를 만들어요.
def softmax(logits: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    s = logits - np.max(logits, axis=1, keepdims=True)
    # 설명: 값을 저장하거나 바꿔요.
    e = np.exp(s)
    # 설명: 함수 결과를 돌려줘요.
    return e / np.sum(e, axis=1, keepdims=True)


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
    # 설명: 다음 코드를 실행해요.
    np.random.seed(7)

    # 설명: 값을 저장하거나 바꿔요.
    X = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [0.2, 0.1, 0.7, 0.0],
            # 설명: 다음 코드를 실행해요.
            [0.9, 0.1, 0.0, 0.3],
            # 설명: 다음 코드를 실행해요.
            [0.3, 0.8, 0.2, 0.1],
            # 설명: 다음 코드를 실행해요.
            [0.8, 0.2, 0.1, 0.4],
            # 설명: 다음 코드를 실행해요.
            [0.1, 0.7, 0.4, 0.2],
            # 설명: 다음 코드를 실행해요.
            [0.9, 0.2, 0.2, 0.8],
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([2, 0, 1, 0, 1, 0])
    # 설명: 값을 저장하거나 바꿔요.
    y_oh = one_hot(y, 3)

    # 설명: 값을 저장하거나 바꿔요.
    W1 = np.random.randn(4, 6) * 0.1
    # 설명: 값을 저장하거나 바꿔요.
    b1 = np.zeros((1, 6))
    # 설명: 값을 저장하거나 바꿔요.
    W2 = np.random.randn(6, 3) * 0.1
    # 설명: 값을 저장하거나 바꿔요.
    b2 = np.zeros((1, 3))

    # 설명: 값을 저장하거나 바꿔요.
    lr = 0.4
    # 설명: 값을 저장하거나 바꿔요.
    losses = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(250):
        # 설명: 값을 저장하거나 바꿔요.
        z1 = X @ W1 + b1
        # 설명: 값을 저장하거나 바꿔요.
        a1 = np.maximum(0, z1)
        # 설명: 값을 저장하거나 바꿔요.
        logits = a1 @ W2 + b2
        # 설명: 값을 저장하거나 바꿔요.
        probs = softmax(logits)

        # 설명: 값을 저장하거나 바꿔요.
        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
        # 설명: 다음 코드를 실행해요.
        losses.append(float(loss))

        # 설명: 값을 저장하거나 바꿔요.
        n = X.shape[0]
        # 설명: 값을 저장하거나 바꿔요.
        dlogits = (probs - y_oh) / n
        # 설명: 값을 저장하거나 바꿔요.
        dW2 = a1.T @ dlogits
        # 설명: 값을 저장하거나 바꿔요.
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        # 설명: 값을 저장하거나 바꿔요.
        da1 = dlogits @ W2.T
        # 설명: 값을 저장하거나 바꿔요.
        dz1 = da1 * (z1 > 0)
        # 설명: 값을 저장하거나 바꿔요.
        dW1 = X.T @ dz1
        # 설명: 값을 저장하거나 바꿔요.
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 설명: 값을 저장하거나 바꿔요.
        W1 -= lr * dW1
        # 설명: 값을 저장하거나 바꿔요.
        b1 -= lr * db1
        # 설명: 값을 저장하거나 바꿔요.
        W2 -= lr * dW2
        # 설명: 값을 저장하거나 바꿔요.
        b2 -= lr * db2

    # 설명: 값을 저장하거나 바꿔요.
    pred = np.argmax(probs, axis=1)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter28",
        # 설명: 다음 코드를 실행해요.
        "topic": "2층 신경망 fitting",
        # 설명: 다음 코드를 실행해요.
        "initial_loss": round(losses[0], 6),
        # 설명: 다음 코드를 실행해요.
        "final_loss": round(losses[-1], 6),
        # 설명: 값을 저장하거나 바꿔요.
        "train_accuracy": round(float(np.mean(pred == y)), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
