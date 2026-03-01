"""풀링 직관 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "풀링은 특징맵을 작게 줄여 중요한 값만 남긴다."
PRACTICE_30MIN = "max pooling을 직접 구현해 요약 결과를 확인한다."


def max_pool2d(feature_map: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    h, w = feature_map.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    out = np.zeros((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            block = feature_map[i * stride : i * stride + size, j * stride : j * stride + size]
            out[i, j] = np.max(block)

    return out


def run() -> dict:
    feature_map = np.array(
        [
            [1, 2, 0, 1],
            [3, 4, 1, 0],
            [0, 2, 5, 6],
            [1, 1, 2, 8],
        ],
        dtype=float,
    )

    pooled = max_pool2d(feature_map, size=2, stride=2)

    return {
        "chapter": "chapter80",
        "topic": "풀링 직관",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "input_shape": list(feature_map.shape),
        "output_shape": list(pooled.shape),
        "feature_map": feature_map.tolist(),
        "pooled": pooled.tolist(),
    }


if __name__ == "__main__":
    print(run())
