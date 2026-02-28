"""CNN 핵심 연산 구현"""
from __future__ import annotations

import numpy as np


def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i : i + kh, j : j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def max_pool2d(feature_map: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    h, w = feature_map.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            block = feature_map[i * stride : i * stride + size, j * stride : j * stride + size]
            out[i, j] = np.max(block)
    return out


def run() -> dict:
    image = np.array(
        [
            [1, 2, 1, 0],
            [0, 1, 3, 1],
            [2, 1, 0, 2],
            [1, 0, 2, 3],
        ],
        dtype=float,
    )
    kernel = np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        dtype=float,
    )

    conv_out = conv2d_valid(image, kernel)
    relu_out = np.maximum(0, conv_out)
    pool_out = max_pool2d(relu_out, size=2, stride=1)

    return {
        "chapter": "chapter30",
        "topic": "CNN 연산",
        "conv_output": conv_out.tolist(),
        "relu_output": relu_out.tolist(),
        "pool_output": pool_out.tolist(),
    }


if __name__ == "__main__":
    print(run())
