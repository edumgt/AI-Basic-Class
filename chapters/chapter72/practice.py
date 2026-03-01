"""역전파 감각 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "복잡한 식도 체인룰로 쪼개면 각 변수의 기여도를 계산할 수 있다."
PRACTICE_30MIN = "간단한 식에서 미분값을 손계산과 수치미분으로 비교한다."


def run() -> dict:
    # y = (a*b + c)^2
    a = 1.5
    b = -2.0
    c = 0.7

    z = a * b + c
    y = z**2

    # chain rule
    dy_dz = 2.0 * z
    dy_da = dy_dz * b
    dy_db = dy_dz * a
    dy_dc = dy_dz

    eps = 1e-6

    def f(aa: float, bb: float, cc: float) -> float:
        return (aa * bb + cc) ** 2

    num_da = (f(a + eps, b, c) - f(a - eps, b, c)) / (2 * eps)
    num_db = (f(a, b + eps, c) - f(a, b - eps, c)) / (2 * eps)
    num_dc = (f(a, b, c + eps) - f(a, b, c - eps)) / (2 * eps)

    return {
        "chapter": "chapter72",
        "topic": "역전파 감각",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "z": round(float(z), 6),
        "y": round(float(y), 6),
        "analytic_grad": {
            "dy_da": round(float(dy_da), 6),
            "dy_db": round(float(dy_db), 6),
            "dy_dc": round(float(dy_dc), 6),
        },
        "numeric_grad": {
            "dy_da": round(float(num_da), 6),
            "dy_db": round(float(num_db), 6),
            "dy_dc": round(float(num_dc), 6),
        },
        "grad_close": bool(
            np.allclose([dy_da, dy_db, dy_dc], [num_da, num_db, num_dc], atol=1e-5)
        ),
    }


if __name__ == "__main__":
    print(run())
