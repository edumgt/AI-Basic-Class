# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""경사하강법 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "경사하강법은 기울기를 따라 손실이 줄어드는 방향으로 조금씩 이동한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "2차 함수의 최소점으로 이동하는 과정을 기록한다."


# 설명: `_loss` 함수를 만들어요.
def _loss(x: float) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return (x - 3.0) ** 2 + 2.0


# 설명: `_grad` 함수를 만들어요.
def _grad(x: float) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return 2.0 * (x - 3.0)


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    lr = 0.2
    # 설명: 값을 저장하거나 바꿔요.
    steps = 15
    # 설명: 값을 저장하거나 바꿔요.
    x = -5.0

    # 설명: 값을 저장하거나 바꿔요.
    trajectory = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for step in range(steps):
        # 설명: 값을 저장하거나 바꿔요.
        g = _grad(x)
        # 설명: 값을 저장하거나 바꿔요.
        x = x - lr * g
        # 설명: 다음 코드를 실행해요.
        trajectory.append(
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "step": step + 1,
                # 설명: 다음 코드를 실행해요.
                "x": round(float(x), 5),
                # 설명: 다음 코드를 실행해요.
                "loss": round(float(_loss(x)), 5),
            # 설명: 다음 코드를 실행해요.
            }
        # 설명: 다음 코드를 실행해요.
        )

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter71",
        # 설명: 다음 코드를 실행해요.
        "topic": "경사하강법",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "learning_rate": lr,
        # 설명: 다음 코드를 실행해요.
        "start_x": -5.0,
        # 설명: 다음 코드를 실행해요.
        "end_x": round(float(x), 6),
        # 설명: 다음 코드를 실행해요.
        "end_loss": round(float(_loss(x)), 6),
        # 설명: 다음 코드를 실행해요.
        "trajectory": trajectory,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
