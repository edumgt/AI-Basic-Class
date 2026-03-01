"""에러 분석 노트 실습 파일"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


LESSON_10MIN = "성능 수치만 보지 말고 어떤 샘플에서 틀렸는지 확인해야 개선 포인트가 보인다."
PRACTICE_30MIN = "오분류 샘플을 표로 뽑아 에러 노트를 만든다."


def run() -> dict:
    X, y = make_classification(
        n_samples=280,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
    )

    model = RandomForestClassifier(n_estimators=160, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    review = X_test.copy().reset_index(drop=True)
    review["true"] = y_test.reset_index(drop=True)
    review["pred"] = pred
    errors = review[review["true"] != review["pred"]].copy()

    return {
        "chapter": "chapter64",
        "topic": "에러 분석 노트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "test_rows": int(len(review)),
        "error_rows": int(len(errors)),
        "error_rate": round(float(len(errors) / len(review)), 4),
        "error_samples": errors.head(5).round(4).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
