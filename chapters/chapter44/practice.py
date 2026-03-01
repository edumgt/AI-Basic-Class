"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LESSON_10MIN = "원본 데이터를 전처리 파이프라인으로 학습 가능한 형태로 바꿀 수 있다."
PRACTICE_30MIN = "결측치 처리 + 인코딩 + 스케일링을 묶어 분류 모델을 만든다."


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 180

    df = pd.DataFrame(
        {
            "study_hours": rng.normal(4.5, 1.1, size=n),
            "sleep_hours": rng.normal(7.0, 0.8, size=n),
            "city": rng.choice(["Seoul", "Busan", "Daegu"], size=n, p=[0.45, 0.35, 0.20]),
        }
    )

    missing_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[missing_idx[:10], "study_hours"] = np.nan
    df.loc[missing_idx[10:], "city"] = np.nan

    target = (
        (df["study_hours"].fillna(df["study_hours"].mean()) * 0.7)
        + (df["sleep_hours"] * 0.3)
        + rng.normal(0, 0.7, size=n)
        > 5.5
    ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.25, random_state=42, stratify=target
    )

    num_cols = ["study_hours", "sleep_hours"]
    cat_cols = ["city"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    transformed_sample = model.named_steps["preprocessor"].transform(X_test.head(3))

    return {
        "chapter": "chapter44",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(acc, 4),
        "transformed_shape_sample": list(transformed_sample.shape),
    }


if __name__ == "__main__":
    print(run())
