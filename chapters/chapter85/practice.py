"""이상탐지 입문 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "정상 패턴에서 많이 벗어난 값을 이상치로 볼 수 있다."
PRACTICE_30MIN = "z-score를 계산해 이상치를 탐지한다."


def run() -> dict:
    values = np.array([10, 11, 10, 12, 11, 10, 9, 10, 11, 45, 10, 9], dtype=float)

    mean = float(values.mean())
    std = float(values.std())
    z = (values - mean) / (std + 1e-9)

    threshold = 2.0
    outlier_idx = np.where(np.abs(z) > threshold)[0]

    return {
        "chapter": "chapter85",
        "topic": "이상탐지 입문",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "z_scores": np.round(z, 4).tolist(),
        "outlier_indices": outlier_idx.astype(int).tolist(),
        "outlier_values": values[outlier_idx].tolist(),
    }


if __name__ == "__main__":
    print(run())
