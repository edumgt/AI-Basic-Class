# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""초기화와 학습 안정성 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "초기 파라미터에 따라 같은 모델도 학습 속도와 안정성이 달라질 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "서로 다른 초기값으로 선형회귀를 학습해 손실 추세를 비교한다."


# 설명: `_train_linear` 함수를 만들어요.
def _train_linear(init_w: float, init_b: float, lr: float = 0.01, epochs: int = 200) -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    x = np.arange(1, 9, dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    y = 2.0 * x + 1.0

    # 설명: 값을 저장하거나 바꿔요.
    w = init_w
    # 설명: 값을 저장하거나 바꿔요.
    b = init_b
    # 설명: 값을 저장하거나 바꿔요.
    losses = []

    # 설명: 값을 저장하거나 바꿔요.
    n = len(x)
    # 설명: 같은 동작을 여러 번 반복해요.
    for epoch in range(epochs):
        # 설명: 값을 저장하거나 바꿔요.
        pred = w * x + b
        # 설명: 값을 저장하거나 바꿔요.
        err = pred - y
        # 설명: 값을 저장하거나 바꿔요.
        loss = float(np.mean(err**2))

        # 설명: 값을 저장하거나 바꿔요.
        dw = float((2.0 / n) * np.sum(err * x))
        # 설명: 값을 저장하거나 바꿔요.
        db = float((2.0 / n) * np.sum(err))

        # 설명: 값을 저장하거나 바꿔요.
        w -= lr * dw
        # 설명: 값을 저장하거나 바꿔요.
        b -= lr * db

        # 설명: 조건이 맞는지 확인해요.
        if epoch < 5 or epoch >= epochs - 5:
            # 설명: 다음 코드를 실행해요.
            losses.append(round(loss, 6))

    # 설명: 값을 저장하거나 바꿔요.
    final_loss = float(np.mean((w * x + b - y) ** 2))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "init": {"w": init_w, "b": init_b},
        # 설명: 다음 코드를 실행해요.
        "final": {"w": round(float(w), 5), "b": round(float(b), 5)},
        # 설명: 다음 코드를 실행해요.
        "final_loss": round(final_loss, 8),
        # 설명: 다음 코드를 실행해요.
        "sampled_losses": losses,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    near_init = _train_linear(init_w=1.8, init_b=0.6)
    # 설명: 값을 저장하거나 바꿔요.
    far_init = _train_linear(init_w=-8.0, init_b=10.0)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter74",
        # 설명: 다음 코드를 실행해요.
        "topic": "초기화와 학습 안정성",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "near_init_result": near_init,
        # 설명: 다음 코드를 실행해요.
        "far_init_result": far_init,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
