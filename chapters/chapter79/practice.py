# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""합성곱 직관 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "필터(커널)를 슬라이딩하면 경계 같은 특징을 강조할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "간단한 edge filter를 이미지에 적용한다."


# 설명: `conv2d_valid` 함수를 만들어요.
def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    h, w = image.shape
    # 설명: 값을 저장하거나 바꿔요.
    kh, kw = kernel.shape
    # 설명: 값을 저장하거나 바꿔요.
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=float)

    # 설명: 같은 동작을 여러 번 반복해요.
    for i in range(out.shape[0]):
        # 설명: 같은 동작을 여러 번 반복해요.
        for j in range(out.shape[1]):
            # 설명: 값을 저장하거나 바꿔요.
            patch = image[i : i + kh, j : j + kw]
            # 설명: 값을 저장하거나 바꿔요.
            out[i, j] = np.sum(patch * kernel)

    # 설명: 함수 결과를 돌려줘요.
    return out


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    image = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [10, 10, 10, 10, 10],
            # 설명: 다음 코드를 실행해요.
            [10, 10, 20, 20, 20],
            # 설명: 다음 코드를 실행해요.
            [10, 10, 20, 80, 80],
            # 설명: 다음 코드를 실행해요.
            [10, 10, 20, 80, 80],
            # 설명: 다음 코드를 실행해요.
            [10, 10, 20, 80, 80],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    edge_kernel = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [-1, -1, -1],
            # 설명: 다음 코드를 실행해요.
            [0, 0, 0],
            # 설명: 다음 코드를 실행해요.
            [1, 1, 1],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    conv_out = conv2d_valid(image, edge_kernel)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter79",
        # 설명: 다음 코드를 실행해요.
        "topic": "합성곱 직관",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "kernel": edge_kernel.tolist(),
        # 설명: 다음 코드를 실행해요.
        "conv_output": conv_out.round(2).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
