# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""날짜 데이터 다루기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "날짜 하나에서 월, 요일, 주말 여부 같은 파생 정보를 만들 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "datetime 변환 후 파생 컬럼을 생성하고 요일별 평균을 구한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 값을 저장하거나 바꿔요.
            "date": pd.date_range("2026-01-01", periods=14, freq="D").astype(str),
            # 설명: 다음 코드를 실행해요.
            "sales": [120, 135, 128, 140, 150, 165, 158, 162, 170, 175, 168, 180, 190, 185],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    df["date"] = pd.to_datetime(df["date"])
    # 설명: 값을 저장하거나 바꿔요.
    df["month"] = df["date"].dt.month
    # 설명: 값을 저장하거나 바꿔요.
    df["day_of_week"] = df["date"].dt.day_name()
    # 설명: 값을 저장하거나 바꿔요.
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    # 설명: 값을 저장하거나 바꿔요.
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # 설명: 값을 저장하거나 바꿔요.
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=weekday_order, ordered=True)

    # 설명: 값을 저장하거나 바꿔요.
    by_weekday = (
        # 설명: 값을 저장하거나 바꿔요.
        df.groupby("day_of_week", observed=False)["sales"].mean().dropna().round(2).to_dict()
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    preview_cols = ["date", "sales", "month", "day_of_week", "is_weekend"]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter40",
        # 설명: 다음 코드를 실행해요.
        "topic": "날짜 데이터 다루기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "derived_columns": ["month", "day_of_week", "is_weekend"],
        # 설명: 다음 코드를 실행해요.
        "weekday_average_sales": by_weekday,
        # 설명: 값을 저장하거나 바꿔요.
        "preview": df[preview_cols].head(7).astype(str).to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
