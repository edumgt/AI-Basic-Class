"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "응용 문제는 데이터 형태(이미지/텍스트/시계열)에 맞는 접근을 선택하는 것이 핵심이다."
PRACTICE_30MIN = "시계열 미니 데모를 선택해 이동평균 기반 간단 리포트를 만든다."


def run() -> dict:
    # 미니 데모 선택: 시계열
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "traffic": [120, 125, 130, 128, 140, 150, 148, 155, 160, 162],
        }
    )

    df["ma_3"] = df["traffic"].rolling(3).mean()
    df["trend"] = (df["traffic"].diff() > 0).map({True: "up", False: "down_or_same"})

    return {
        "chapter": "chapter88",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "selected_domain": "time_series",
        "last_traffic": int(df["traffic"].iloc[-1]),
        "last_ma_3": round(float(df["ma_3"].iloc[-1]), 4),
        "preview": df.tail(5).astype(str).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
