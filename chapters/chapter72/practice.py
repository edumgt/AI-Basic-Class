# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""역전파 감각 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "복잡한 식도 체인룰로 쪼개면 각 변수의 기여도를 계산할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "간단한 식에서 미분값을 손계산과 수치미분으로 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # y = (a*b + c)^2
    # 설명: 값을 저장하거나 바꿔요.
    a = 1.5
    # 설명: 값을 저장하거나 바꿔요.
    b = -2.0
    # 설명: 값을 저장하거나 바꿔요.
    c = 0.7

    # 설명: 값을 저장하거나 바꿔요.
    z = a * b + c
    # 설명: 값을 저장하거나 바꿔요.
    y = z**2

    # chain rule
    # 설명: 값을 저장하거나 바꿔요.
    dy_dz = 2.0 * z
    # 설명: 값을 저장하거나 바꿔요.
    dy_da = dy_dz * b
    # 설명: 값을 저장하거나 바꿔요.
    dy_db = dy_dz * a
    # 설명: 값을 저장하거나 바꿔요.
    dy_dc = dy_dz

    # 설명: 값을 저장하거나 바꿔요.
    eps = 1e-6

    # 설명: `f` 함수를 만들어요.
    def f(aa: float, bb: float, cc: float) -> float:
        # 설명: 함수 결과를 돌려줘요.
        return (aa * bb + cc) ** 2

    # 설명: 값을 저장하거나 바꿔요.
    num_da = (f(a + eps, b, c) - f(a - eps, b, c)) / (2 * eps)
    # 설명: 값을 저장하거나 바꿔요.
    num_db = (f(a, b + eps, c) - f(a, b - eps, c)) / (2 * eps)
    # 설명: 값을 저장하거나 바꿔요.
    num_dc = (f(a, b, c + eps) - f(a, b, c - eps)) / (2 * eps)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter72",
        # 설명: 다음 코드를 실행해요.
        "topic": "역전파 감각",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "z": round(float(z), 6),
        # 설명: 다음 코드를 실행해요.
        "y": round(float(y), 6),
        # 설명: 다음 코드를 실행해요.
        "analytic_grad": {
            # 설명: 다음 코드를 실행해요.
            "dy_da": round(float(dy_da), 6),
            # 설명: 다음 코드를 실행해요.
            "dy_db": round(float(dy_db), 6),
            # 설명: 다음 코드를 실행해요.
            "dy_dc": round(float(dy_dc), 6),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "numeric_grad": {
            # 설명: 다음 코드를 실행해요.
            "dy_da": round(float(num_da), 6),
            # 설명: 다음 코드를 실행해요.
            "dy_db": round(float(num_db), 6),
            # 설명: 다음 코드를 실행해요.
            "dy_dc": round(float(num_dc), 6),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "grad_close": bool(
            # 설명: 값을 저장하거나 바꿔요.
            np.allclose([dy_da, dy_db, dy_dc], [num_da, num_db, num_dc], atol=1e-5)
        # 설명: 다음 코드를 실행해요.
        ),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
