"""합성곱 직관 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "필터(커널)를 슬라이딩하면 경계 같은 특징을 강조할 수 있다."
PRACTICE_30MIN = "간단한 edge filter를 이미지에 적용한다."


def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=float)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i : i + kh, j : j + kw]
            out[i, j] = np.sum(patch * kernel)

    return out


def run() -> dict:
    image = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 10, 20, 20, 20],
            [10, 10, 20, 80, 80],
            [10, 10, 20, 80, 80],
            [10, 10, 20, 80, 80],
        ],
        dtype=float,
    )

    edge_kernel = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=float,
    )

    conv_out = conv2d_valid(image, edge_kernel)

    return {
        "chapter": "chapter79",
        "topic": "합성곱 직관",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "kernel": edge_kernel.tolist(),
        "conv_output": conv_out.round(2).tolist(),
    }


if __name__ == "__main__":
    print(run())
