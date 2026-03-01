# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""풀링 직관 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "풀링은 특징맵을 작게 줄여 중요한 값만 남긴다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "max pooling을 직접 구현해 요약 결과를 확인한다."


# 설명: `max_pool2d` 함수를 만들어요.
def max_pool2d(feature_map: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    # 설명: 값을 저장하거나 바꿔요.
    h, w = feature_map.shape
    # 설명: 값을 저장하거나 바꿔요.
    out_h = (h - size) // stride + 1
    # 설명: 값을 저장하거나 바꿔요.
    out_w = (w - size) // stride + 1
    # 설명: 값을 저장하거나 바꿔요.
    out = np.zeros((out_h, out_w), dtype=float)

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
    feature_map = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [1, 2, 0, 1],
            # 설명: 다음 코드를 실행해요.
            [3, 4, 1, 0],
            # 설명: 다음 코드를 실행해요.
            [0, 2, 5, 6],
            # 설명: 다음 코드를 실행해요.
            [1, 1, 2, 8],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    pooled = max_pool2d(feature_map, size=2, stride=2)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter80",
        # 설명: 다음 코드를 실행해요.
        "topic": "풀링 직관",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "input_shape": list(feature_map.shape),
        # 설명: 다음 코드를 실행해요.
        "output_shape": list(pooled.shape),
        # 설명: 다음 코드를 실행해요.
        "feature_map": feature_map.tolist(),
        # 설명: 다음 코드를 실행해요.
        "pooled": pooled.tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
