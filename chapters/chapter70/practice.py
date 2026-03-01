# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""손실함수 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "손실함수는 예측이 얼마나 틀렸는지 숫자로 알려준다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "MSE와 Cross-Entropy를 직접 계산한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 회귀 예시
    # 설명: 값을 저장하거나 바꿔요.
    y_true_reg = np.array([3.0, 5.0, 2.5, 7.0], dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    y_pred_reg = np.array([2.8, 4.6, 3.1, 6.2], dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    mse = float(np.mean((y_true_reg - y_pred_reg) ** 2))

    # 분류 예시(3클래스)
    # 설명: 값을 저장하거나 바꿔요.
    y_true_cls = np.array([0, 2, 1], dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    y_prob = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [0.80, 0.15, 0.05],
            # 설명: 다음 코드를 실행해요.
            [0.10, 0.20, 0.70],
            # 설명: 다음 코드를 실행해요.
            [0.25, 0.60, 0.15],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    eps = 1e-9
    # 설명: 값을 저장하거나 바꿔요.
    chosen_prob = y_prob[np.arange(len(y_true_cls)), y_true_cls]
    # 설명: 값을 저장하거나 바꿔요.
    cross_entropy = float(-np.mean(np.log(chosen_prob + eps)))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter70",
        # 설명: 다음 코드를 실행해요.
        "topic": "손실함수",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "mse": round(mse, 6),
        # 설명: 다음 코드를 실행해요.
        "cross_entropy": round(cross_entropy, 6),
        # 설명: 다음 코드를 실행해요.
        "chosen_probabilities": chosen_prob.round(4).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
