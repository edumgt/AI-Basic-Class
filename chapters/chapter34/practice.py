"""문자열 숫자로 바꾸기 실습 파일"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


LESSON_10MIN = "모델은 텍스트를 직접 이해하지 못하므로 숫자로 변환해야 한다."
PRACTICE_30MIN = "라벨 인코딩으로 도시 이름을 정수로 바꾼다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "city": ["Seoul", "Busan", "Daegu", "Seoul", "Busan", "Incheon"],
            "score": [88, 76, 91, 84, 79, 90],
        }
    )

    categories = sorted(df["city"].unique())
    manual_map = {name: idx for idx, name in enumerate(categories)}
    df["city_manual_encoded"] = df["city"].map(manual_map)

    encoder = LabelEncoder()
    df["city_label_encoded"] = encoder.fit_transform(df["city"])

    return {
        "chapter": "chapter34",
        "topic": "문자열 숫자로 바꾸기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "manual_mapping": manual_map,
        "label_encoder_classes": encoder.classes_.tolist(),
        "encoded_preview": df.head(6).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
