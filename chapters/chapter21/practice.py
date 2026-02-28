"""신경망 기초(순전파/역전파/경사하강법) 실습 파일"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def run() -> dict:
    np.random.seed(42)

    # 입력 4개, 은닉 5개, 출력 클래스 3개인 작은 신경망
    X = np.array([
        [0.2, 0.1, 0.7, 0.0],
        [0.9, 0.1, 0.0, 0.3],
        [0.3, 0.8, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.4],
        [0.1, 0.7, 0.4, 0.2],
        [0.9, 0.2, 0.2, 0.8],
    ])
    y = np.array([2, 0, 1, 0, 1, 0])

    # 가중치 행렬 (weight matrix)
    W1 = np.random.randn(4, 5) * 0.1  # (input_dim, hidden_dim)
    b1 = np.zeros((1, 5))
    W2 = np.random.randn(5, 3) * 0.1  # (hidden_dim, num_classes)
    b2 = np.zeros((1, 3))

    y_one_hot = one_hot(y, num_classes=3)

    lr = 0.5
    epochs = 300
    losses = []

    for _ in range(epochs):
        # 1) 순전파(Forward Propagation)
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU
        logits = a1 @ W2 + b2
        probs = softmax(logits)

        # 2) 크로스 엔트로피 손실
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-12), axis=1))
        losses.append(float(loss))

        # 3) 역전파(Backward Propagation)
        n = X.shape[0]
        dlogits = (probs - y_one_hot) / n
        dW2 = a1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        da1 = dlogits @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 4) 경사하강법(Gradient Descent) 업데이트
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y))

    # CNN 핵심 아이디어: 필터와 입력의 element-wise 곱 + 합
    image_patch = np.array(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]]
    )
    kernel = np.array(
        [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
    )
    conv_value = float(np.sum(image_patch * kernel))

    return {
        "chapter": "chapter21",
        "topic": "신경망 기초와 학습",
        "weight_shapes": {
            "W1": list(W1.shape),
            "b1": list(b1.shape),
            "W2": list(W2.shape),
            "b2": list(b2.shape),
        },
        "fitting_epochs": epochs,
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "train_accuracy": round(accuracy, 4),
        "softmax_example": np.round(probs[0], 4).tolist(),
        "cnn_conv_example": conv_value,
    }


if __name__ == "__main__":
    print(run())
