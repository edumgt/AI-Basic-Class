"""중복과 오탈자 정리 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "같은 데이터 중복과 오탈자를 정리하면 분석 품질이 올라간다."
PRACTICE_30MIN = "replace와 drop_duplicates로 데이터 품질을 개선한다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "name": ["Mina", "Mina", "Joon", "Sara", "Sara", "Noah", "Noah"],
            "city": ["Seol", "Seol", "Busn", "Seoul", "Seoul", "Inchon", "Incheon"],
            "score": [91, 91, 84, 88, 88, 79, 79],
        }
    )

    typo_map = {
        "Seol": "Seoul",
        "Busn": "Busan",
        "Inchon": "Incheon",
    }

    cleaned = df.copy()
    cleaned["city"] = cleaned["city"].replace(typo_map)
    cleaned = cleaned.drop_duplicates(subset=["name", "city", "score"], keep="first")

    return {
        "chapter": "chapter39",
        "topic": "중복/오탈자 정리",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "rows_before": int(len(df)),
        "rows_after": int(len(cleaned)),
        "city_values_before": sorted(df["city"].unique().tolist()),
        "city_values_after": sorted(cleaned["city"].unique().tolist()),
        "cleaned_preview": cleaned.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
