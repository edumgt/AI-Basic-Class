# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""중복과 오탈자 정리 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "같은 데이터 중복과 오탈자를 정리하면 분석 품질이 올라간다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "replace와 drop_duplicates로 데이터 품질을 개선한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "name": ["Mina", "Mina", "Joon", "Sara", "Sara", "Noah", "Noah"],
            # 설명: 다음 코드를 실행해요.
            "city": ["Seol", "Seol", "Busn", "Seoul", "Seoul", "Inchon", "Incheon"],
            # 설명: 다음 코드를 실행해요.
            "score": [91, 91, 84, 88, 88, 79, 79],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    typo_map = {
        # 설명: 다음 코드를 실행해요.
        "Seol": "Seoul",
        # 설명: 다음 코드를 실행해요.
        "Busn": "Busan",
        # 설명: 다음 코드를 실행해요.
        "Inchon": "Incheon",
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    cleaned = df.copy()
    # 설명: 값을 저장하거나 바꿔요.
    cleaned["city"] = cleaned["city"].replace(typo_map)
    # 설명: 값을 저장하거나 바꿔요.
    cleaned = cleaned.drop_duplicates(subset=["name", "city", "score"], keep="first")

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter39",
        # 설명: 다음 코드를 실행해요.
        "topic": "중복/오탈자 정리",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "rows_before": int(len(df)),
        # 설명: 다음 코드를 실행해요.
        "rows_after": int(len(cleaned)),
        # 설명: 다음 코드를 실행해요.
        "city_values_before": sorted(df["city"].unique().tolist()),
        # 설명: 다음 코드를 실행해요.
        "city_values_after": sorted(cleaned["city"].unique().tolist()),
        # 설명: 값을 저장하거나 바꿔요.
        "cleaned_preview": cleaned.to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
