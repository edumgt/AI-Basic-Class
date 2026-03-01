# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""원-핫 인코딩 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "범주형 값은 0/1 깃발 컬럼으로 안전하게 표현할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "get_dummies로 색상과 동물 컬럼을 원-핫 인코딩한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "color": ["red", "blue", "green", "blue", "red"],
            # 설명: 다음 코드를 실행해요.
            "animal": ["cat", "dog", "cat", "bird", "dog"],
            # 설명: 다음 코드를 실행해요.
            "value": [10, 20, 12, 18, 16],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    encoded = pd.get_dummies(df, columns=["color", "animal"], dtype=int)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter35",
        # 설명: 다음 코드를 실행해요.
        "topic": "원-핫 인코딩",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "original_columns": df.columns.tolist(),
        # 설명: 다음 코드를 실행해요.
        "encoded_columns": encoded.columns.tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "encoded_preview": encoded.to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
