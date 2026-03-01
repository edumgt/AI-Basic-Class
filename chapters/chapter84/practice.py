"""시계열 입문 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "시계열 데이터는 시간 순서를 보존하고 흐름(추세)을 관찰해야 한다."
PRACTICE_30MIN = "이동평균으로 일별 값의 노이즈를 줄여 추세를 본다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-02-01", periods=12, freq="D"),
            "value": [10, 11, 13, 12, 15, 16, 18, 17, 19, 21, 20, 22],
        }
    )

    df["ma_3"] = df["value"].rolling(window=3).mean()
    df["diff_1"] = df["value"].diff()

    return {
        "chapter": "chapter84",
        "topic": "시계열 입문",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "preview": df.head(8).round(3).astype(str).to_dict(orient="records"),
        "last_ma_3": round(float(df["ma_3"].iloc[-1]), 4),
    }


if __name__ == "__main__":
    print(run())
