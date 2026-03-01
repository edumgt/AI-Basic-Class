# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""문자열 숫자로 바꾸기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import LabelEncoder


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "모델은 텍스트를 직접 이해하지 못하므로 숫자로 변환해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "라벨 인코딩으로 도시 이름을 정수로 바꾼다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "city": ["Seoul", "Busan", "Daegu", "Seoul", "Busan", "Incheon"],
            # 설명: 다음 코드를 실행해요.
            "score": [88, 76, 91, 84, 79, 90],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    categories = sorted(df["city"].unique())
    # 설명: 값을 저장하거나 바꿔요.
    manual_map = {name: idx for idx, name in enumerate(categories)}
    # 설명: 값을 저장하거나 바꿔요.
    df["city_manual_encoded"] = df["city"].map(manual_map)

    # 설명: 값을 저장하거나 바꿔요.
    encoder = LabelEncoder()
    # 설명: 값을 저장하거나 바꿔요.
    df["city_label_encoded"] = encoder.fit_transform(df["city"])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter34",
        # 설명: 다음 코드를 실행해요.
        "topic": "문자열 숫자로 바꾸기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "manual_mapping": manual_map,
        # 설명: 다음 코드를 실행해요.
        "label_encoder_classes": encoder.classes_.tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "encoded_preview": df.head(6).to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
