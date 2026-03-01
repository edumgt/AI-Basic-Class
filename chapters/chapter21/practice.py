# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""신경망 기초(순전파/역전파/경사하강법) 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `softmax` 함수를 만들어요.
def softmax(logits: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    # 설명: 값을 저장하거나 바꿔요.
    exp_scores = np.exp(shifted)
    # 설명: 함수 결과를 돌려줘요.
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 설명: `one_hot` 함수를 만들어요.
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    encoded = np.zeros((len(y), num_classes))
    # 설명: 값을 저장하거나 바꿔요.
    encoded[np.arange(len(y)), y] = 1.0
    # 설명: 함수 결과를 돌려줘요.
    return encoded


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 다음 코드를 실행해요.
    np.random.seed(42)

    # 입력 4개, 은닉 5개, 출력 클래스 3개인 작은 신경망
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array([
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
    ])
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([2, 0, 1, 0, 1, 0])

    # 가중치 행렬 (weight matrix)
    # 설명: 값을 저장하거나 바꿔요.
    W1 = np.random.randn(4, 5) * 0.1  # (input_dim, hidden_dim)
    # 설명: 값을 저장하거나 바꿔요.
    b1 = np.zeros((1, 5))
    # 설명: 값을 저장하거나 바꿔요.
    W2 = np.random.randn(5, 3) * 0.1  # (hidden_dim, num_classes)
    # 설명: 값을 저장하거나 바꿔요.
    b2 = np.zeros((1, 3))

    # 설명: 값을 저장하거나 바꿔요.
    y_one_hot = one_hot(y, num_classes=3)

    # 설명: 값을 저장하거나 바꿔요.
    lr = 0.5
    # 설명: 값을 저장하거나 바꿔요.
    epochs = 300
    # 설명: 값을 저장하거나 바꿔요.
    losses = []

    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(epochs):
        # 1) 순전파(Forward Propagation)
        # 설명: 값을 저장하거나 바꿔요.
        z1 = X @ W1 + b1
        # 설명: 값을 저장하거나 바꿔요.
        a1 = np.maximum(0, z1)  # ReLU
        # 설명: 값을 저장하거나 바꿔요.
        logits = a1 @ W2 + b2
        # 설명: 값을 저장하거나 바꿔요.
        probs = softmax(logits)

        # 2) 크로스 엔트로피 손실
        # 설명: 값을 저장하거나 바꿔요.
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-12), axis=1))
        # 설명: 다음 코드를 실행해요.
        losses.append(float(loss))

        # 3) 역전파(Backward Propagation)
        # 설명: 값을 저장하거나 바꿔요.
        n = X.shape[0]
        # 설명: 값을 저장하거나 바꿔요.
        dlogits = (probs - y_one_hot) / n
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

        # 4) 경사하강법(Gradient Descent) 업데이트
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
    # 설명: 값을 저장하거나 바꿔요.
    accuracy = float(np.mean(pred == y))

    # CNN 핵심 아이디어: 필터와 입력의 element-wise 곱 + 합
    # 설명: 값을 저장하거나 바꿔요.
    image_patch = np.array(
        # 설명: 다음 코드를 실행해요.
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]]
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    kernel = np.array(
        # 설명: 다음 코드를 실행해요.
        [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    conv_value = float(np.sum(image_patch * kernel))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter21",
        # 설명: 다음 코드를 실행해요.
        "topic": "신경망 기초와 학습",
        # 설명: 다음 코드를 실행해요.
        "weight_shapes": {
            # 설명: 다음 코드를 실행해요.
            "W1": list(W1.shape),
            # 설명: 다음 코드를 실행해요.
            "b1": list(b1.shape),
            # 설명: 다음 코드를 실행해요.
            "W2": list(W2.shape),
            # 설명: 다음 코드를 실행해요.
            "b2": list(b2.shape),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "fitting_epochs": epochs,
        # 설명: 다음 코드를 실행해요.
        "initial_loss": round(losses[0], 6),
        # 설명: 다음 코드를 실행해요.
        "final_loss": round(losses[-1], 6),
        # 설명: 다음 코드를 실행해요.
        "train_accuracy": round(accuracy, 4),
        # 설명: 다음 코드를 실행해요.
        "softmax_example": np.round(probs[0], 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "cnn_conv_example": conv_value,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
