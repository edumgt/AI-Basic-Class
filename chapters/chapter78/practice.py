"""이미지 데이터 입문 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "흑백 이미지는 2차원 숫자 배열(픽셀 밝기)로 표현된다."
PRACTICE_30MIN = "numpy 배열 연산으로 반전/밝기 조절을 수행한다."


def run() -> dict:
    image = np.array(
        [
            [10, 30, 60, 90, 120],
            [20, 40, 80, 110, 130],
            [30, 60, 100, 140, 160],
            [40, 70, 120, 170, 200],
            [50, 80, 140, 190, 230],
        ],
        dtype=np.uint8,
    )

    inverted = 255 - image
    brighter = np.clip(image + 40, 0, 255).astype(np.uint8)

    return {
        "chapter": "chapter78",
        "topic": "이미지 데이터 입문",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "shape": list(image.shape),
        "original": image.tolist(),
        "inverted": inverted.tolist(),
        "brighter": brighter.tolist(),
    }


if __name__ == "__main__":
    print(run())
