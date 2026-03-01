"""날짜 데이터 다루기 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "날짜 하나에서 월, 요일, 주말 여부 같은 파생 정보를 만들 수 있다."
PRACTICE_30MIN = "datetime 변환 후 파생 컬럼을 생성하고 요일별 평균을 구한다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=14, freq="D").astype(str),
            "sales": [120, 135, 128, 140, 150, 165, 158, 162, 170, 175, 168, 180, 190, 185],
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=weekday_order, ordered=True)

    by_weekday = (
        df.groupby("day_of_week", observed=False)["sales"].mean().dropna().round(2).to_dict()
    )

    preview_cols = ["date", "sales", "month", "day_of_week", "is_weekend"]

    return {
        "chapter": "chapter40",
        "topic": "날짜 데이터 다루기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "derived_columns": ["month", "day_of_week", "is_weekend"],
        "weekday_average_sales": by_weekday,
        "preview": df[preview_cols].head(7).astype(str).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
