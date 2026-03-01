# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""이상탐지 입문 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "정상 패턴에서 많이 벗어난 값을 이상치로 볼 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "z-score를 계산해 이상치를 탐지한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    values = np.array([10, 11, 10, 12, 11, 10, 9, 10, 11, 45, 10, 9], dtype=float)

    # 설명: 값을 저장하거나 바꿔요.
    mean = float(values.mean())
    # 설명: 값을 저장하거나 바꿔요.
    std = float(values.std())
    # 설명: 값을 저장하거나 바꿔요.
    z = (values - mean) / (std + 1e-9)

    # 설명: 값을 저장하거나 바꿔요.
    threshold = 2.0
    # 설명: 값을 저장하거나 바꿔요.
    outlier_idx = np.where(np.abs(z) > threshold)[0]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter85",
        # 설명: 다음 코드를 실행해요.
        "topic": "이상탐지 입문",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "mean": round(mean, 4),
        # 설명: 다음 코드를 실행해요.
        "std": round(std, 4),
        # 설명: 다음 코드를 실행해요.
        "z_scores": np.round(z, 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "outlier_indices": outlier_idx.astype(int).tolist(),
        # 설명: 다음 코드를 실행해요.
        "outlier_values": values[outlier_idx].tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
