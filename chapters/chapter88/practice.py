# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "응용 문제는 데이터 형태(이미지/텍스트/시계열)에 맞는 접근을 선택하는 것이 핵심이다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "시계열 미니 데모를 선택해 이동평균 기반 간단 리포트를 만든다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 미니 데모 선택: 시계열
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 값을 저장하거나 바꿔요.
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            # 설명: 다음 코드를 실행해요.
            "traffic": [120, 125, 130, 128, 140, 150, 148, 155, 160, 162],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    df["ma_3"] = df["traffic"].rolling(3).mean()
    # 설명: 값을 저장하거나 바꿔요.
    df["trend"] = (df["traffic"].diff() > 0).map({True: "up", False: "down_or_same"})

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter88",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "selected_domain": "time_series",
        # 설명: 다음 코드를 실행해요.
        "last_traffic": int(df["traffic"].iloc[-1]),
        # 설명: 다음 코드를 실행해요.
        "last_ma_3": round(float(df["ma_3"].iloc[-1]), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "preview": df.tail(5).astype(str).to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
