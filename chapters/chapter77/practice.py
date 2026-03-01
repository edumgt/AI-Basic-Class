# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "순전파와 역전파를 반복하면 작은 신경망도 점점 오차를 줄일 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "numpy로 2층 신경망을 학습해 XOR 예제를 맞춘다."


# 설명: `_sigmoid` 함수를 만들어요.
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # 설명: 함수 결과를 돌려줘요.
    return 1.0 / (1.0 + np.exp(-x))


# 설명: `_bce_loss` 함수를 만들어요.
def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 설명: 값을 저장하거나 바꿔요.
    eps = 1e-9
    # 설명: 함수 결과를 돌려줘요.
    return float(-np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # XOR 데이터
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [0.0, 0.0],
            # 설명: 다음 코드를 실행해요.
            [0.0, 1.0],
            # 설명: 다음 코드를 실행해요.
            [1.0, 0.0],
            # 설명: 다음 코드를 실행해요.
            [1.0, 1.0],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)

    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    hidden_dim = 3

    # 설명: 값을 저장하거나 바꿔요.
    W1 = rng.normal(0, 0.5, size=(2, hidden_dim))
    # 설명: 값을 저장하거나 바꿔요.
    b1 = np.zeros((1, hidden_dim), dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    W2 = rng.normal(0, 0.5, size=(hidden_dim, 1))
    # 설명: 값을 저장하거나 바꿔요.
    b2 = np.zeros((1, 1), dtype=float)

    # 설명: 값을 저장하거나 바꿔요.
    lr = 0.8
    # 설명: 값을 저장하거나 바꿔요.
    epochs = 800

    # 초기 손실
    # 설명: 값을 저장하거나 바꿔요.
    a1_init = _sigmoid(X @ W1 + b1)
    # 설명: 값을 저장하거나 바꿔요.
    y_hat_init = _sigmoid(a1_init @ W2 + b2)
    # 설명: 값을 저장하거나 바꿔요.
    loss_before = _bce_loss(y, y_hat_init)

    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(epochs):
        # 설명: 값을 저장하거나 바꿔요.
        z1 = X @ W1 + b1
        # 설명: 값을 저장하거나 바꿔요.
        a1 = _sigmoid(z1)
        # 설명: 값을 저장하거나 바꿔요.
        z2 = a1 @ W2 + b2
        # 설명: 값을 저장하거나 바꿔요.
        y_hat = _sigmoid(z2)

        # BCE + sigmoid gradient
        # 설명: 값을 저장하거나 바꿔요.
        n = len(X)
        # 설명: 값을 저장하거나 바꿔요.
        dz2 = (y_hat - y) / n
        # 설명: 값을 저장하거나 바꿔요.
        dW2 = a1.T @ dz2
        # 설명: 값을 저장하거나 바꿔요.
        db2 = dz2.sum(axis=0, keepdims=True)

        # 설명: 값을 저장하거나 바꿔요.
        da1 = dz2 @ W2.T
        # 설명: 값을 저장하거나 바꿔요.
        dz1 = da1 * a1 * (1.0 - a1)
        # 설명: 값을 저장하거나 바꿔요.
        dW1 = X.T @ dz1
        # 설명: 값을 저장하거나 바꿔요.
        db1 = dz1.sum(axis=0, keepdims=True)

        # 설명: 값을 저장하거나 바꿔요.
        W2 -= lr * dW2
        # 설명: 값을 저장하거나 바꿔요.
        b2 -= lr * db2
        # 설명: 값을 저장하거나 바꿔요.
        W1 -= lr * dW1
        # 설명: 값을 저장하거나 바꿔요.
        b1 -= lr * db1

    # 설명: 값을 저장하거나 바꿔요.
    a1_final = _sigmoid(X @ W1 + b1)
    # 설명: 값을 저장하거나 바꿔요.
    y_hat_final = _sigmoid(a1_final @ W2 + b2)
    # 설명: 값을 저장하거나 바꿔요.
    loss_after = _bce_loss(y, y_hat_final)

    # 설명: 값을 저장하거나 바꿔요.
    pred = (y_hat_final >= 0.5).astype(int)
    # 설명: 값을 저장하거나 바꿔요.
    acc = float((pred == y).mean())

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter77",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "loss_before": round(loss_before, 6),
        # 설명: 다음 코드를 실행해요.
        "loss_after": round(loss_after, 6),
        # 설명: 다음 코드를 실행해요.
        "predictions": pred.flatten().tolist(),
        # 설명: 다음 코드를 실행해요.
        "target": y.astype(int).flatten().tolist(),
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(acc, 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
