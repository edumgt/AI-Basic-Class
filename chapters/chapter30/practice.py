# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""CNN 핵심 연산 구현"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: `conv2d_valid` 함수를 만들어요.
def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    h, w = image.shape
    # 설명: 값을 저장하거나 바꿔요.
    kh, kw = kernel.shape
    # 설명: 값을 저장하거나 바꿔요.
    out = np.zeros((h - kh + 1, w - kw + 1))
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


# 설명: `max_pool2d` 함수를 만들어요.
def max_pool2d(feature_map: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    h, w = feature_map.shape
    # 설명: 값을 저장하거나 바꿔요.
    out_h = (h - size) // stride + 1
    # 설명: 값을 저장하거나 바꿔요.
    out_w = (w - size) // stride + 1
    # 설명: 값을 저장하거나 바꿔요.
    out = np.zeros((out_h, out_w))
    # 설명: 같은 동작을 여러 번 반복해요.
    for i in range(out_h):
        # 설명: 같은 동작을 여러 번 반복해요.
        for j in range(out_w):
            # 설명: 값을 저장하거나 바꿔요.
            block = feature_map[i * stride : i * stride + size, j * stride : j * stride + size]
            # 설명: 값을 저장하거나 바꿔요.
            out[i, j] = np.max(block)
    # 설명: 함수 결과를 돌려줘요.
    return out


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    image = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [1, 2, 1, 0],
            # 설명: 다음 코드를 실행해요.
            [0, 1, 3, 1],
            # 설명: 다음 코드를 실행해요.
            [2, 1, 0, 2],
            # 설명: 다음 코드를 실행해요.
            [1, 0, 2, 3],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    kernel = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [1, 0, -1],
            # 설명: 다음 코드를 실행해요.
            [1, 0, -1],
            # 설명: 다음 코드를 실행해요.
            [1, 0, -1],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    conv_out = conv2d_valid(image, kernel)
    # 설명: 값을 저장하거나 바꿔요.
    relu_out = np.maximum(0, conv_out)
    # 설명: 값을 저장하거나 바꿔요.
    pool_out = max_pool2d(relu_out, size=2, stride=1)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter30",
        # 설명: 다음 코드를 실행해요.
        "topic": "CNN 연산",
        # 설명: 다음 코드를 실행해요.
        "conv_output": conv_out.tolist(),
        # 설명: 다음 코드를 실행해요.
        "relu_output": relu_out.tolist(),
        # 설명: 다음 코드를 실행해요.
        "pool_output": pool_out.tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
