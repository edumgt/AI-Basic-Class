"""원-핫 인코딩 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "범주형 값은 0/1 깃발 컬럼으로 안전하게 표현할 수 있다."
PRACTICE_30MIN = "get_dummies로 색상과 동물 컬럼을 원-핫 인코딩한다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "blue", "red"],
            "animal": ["cat", "dog", "cat", "bird", "dog"],
            "value": [10, 20, 12, 18, 16],
        }
    )

    encoded = pd.get_dummies(df, columns=["color", "animal"], dtype=int)

    return {
        "chapter": "chapter35",
        "topic": "원-핫 인코딩",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "original_columns": df.columns.tolist(),
        "encoded_columns": encoded.columns.tolist(),
        "encoded_preview": encoded.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
