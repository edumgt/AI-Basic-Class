# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""활성화 함수 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "활성화 함수는 직선만으로는 못 배우는 복잡한 패턴을 학습하게 돕는다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "ReLU, Sigmoid, Tanh 값을 같은 입력에서 비교한다."


# 설명: `_sigmoid` 함수를 만들어요.
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # 설명: 함수 결과를 돌려줘요.
    return 1.0 / (1.0 + np.exp(-x))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    x = np.linspace(-3, 3, 13)
    # 설명: 값을 저장하거나 바꿔요.
    relu = np.maximum(0.0, x)
    # 설명: 값을 저장하거나 바꿔요.
    sigmoid = _sigmoid(x)
    # 설명: 값을 저장하거나 바꿔요.
    tanh = np.tanh(x)

    # 설명: 값을 저장하거나 바꿔요.
    sample_idx = [2, 4, 6, 8, 10]
    # 설명: 값을 저장하거나 바꿔요.
    table = [
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "x": round(float(x[i]), 2),
            # 설명: 다음 코드를 실행해요.
            "relu": round(float(relu[i]), 4),
            # 설명: 다음 코드를 실행해요.
            "sigmoid": round(float(sigmoid[i]), 4),
            # 설명: 다음 코드를 실행해요.
            "tanh": round(float(tanh[i]), 4),
        # 설명: 다음 코드를 실행해요.
        }
        # 설명: 같은 동작을 여러 번 반복해요.
        for i in sample_idx
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter68",
        # 설명: 다음 코드를 실행해요.
        "topic": "활성화 함수",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "x_values": x.round(2).tolist(),
        # 설명: 다음 코드를 실행해요.
        "relu": relu.round(4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "sigmoid": sigmoid.round(4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "tanh": tanh.round(4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "sample_table": table,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
