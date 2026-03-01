# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""시계열 입문 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "시계열 데이터는 시간 순서를 보존하고 흐름(추세)을 관찰해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "이동평균으로 일별 값의 노이즈를 줄여 추세를 본다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 값을 저장하거나 바꿔요.
            "date": pd.date_range("2026-02-01", periods=12, freq="D"),
            # 설명: 다음 코드를 실행해요.
            "value": [10, 11, 13, 12, 15, 16, 18, 17, 19, 21, 20, 22],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    df["ma_3"] = df["value"].rolling(window=3).mean()
    # 설명: 값을 저장하거나 바꿔요.
    df["diff_1"] = df["value"].diff()

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter84",
        # 설명: 다음 코드를 실행해요.
        "topic": "시계열 입문",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 값을 저장하거나 바꿔요.
        "preview": df.head(8).round(3).astype(str).to_dict(orient="records"),
        # 설명: 다음 코드를 실행해요.
        "last_ma_3": round(float(df["ma_3"].iloc[-1]), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
