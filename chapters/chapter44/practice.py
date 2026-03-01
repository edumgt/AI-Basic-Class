# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.compose import ColumnTransformer
# 설명: 필요한 도구를 가져와요.
from sklearn.impute import SimpleImputer
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "원본 데이터를 전처리 파이프라인으로 학습 가능한 형태로 바꿀 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "결측치 처리 + 인코딩 + 스케일링을 묶어 분류 모델을 만든다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 180

    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 값을 저장하거나 바꿔요.
            "study_hours": rng.normal(4.5, 1.1, size=n),
            # 설명: 값을 저장하거나 바꿔요.
            "sleep_hours": rng.normal(7.0, 0.8, size=n),
            # 설명: 값을 저장하거나 바꿔요.
            "city": rng.choice(["Seoul", "Busan", "Daegu"], size=n, p=[0.45, 0.35, 0.20]),
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    missing_idx = rng.choice(df.index, size=20, replace=False)
    # 설명: 값을 저장하거나 바꿔요.
    df.loc[missing_idx[:10], "study_hours"] = np.nan
    # 설명: 값을 저장하거나 바꿔요.
    df.loc[missing_idx[10:], "city"] = np.nan

    # 설명: 값을 저장하거나 바꿔요.
    target = (
        # 설명: 다음 코드를 실행해요.
        (df["study_hours"].fillna(df["study_hours"].mean()) * 0.7)
        # 설명: 다음 코드를 실행해요.
        + (df["sleep_hours"] * 0.3)
        # 설명: 값을 저장하거나 바꿔요.
        + rng.normal(0, 0.7, size=n)
        # 설명: 다음 코드를 실행해요.
        > 5.5
    # 설명: 다음 코드를 실행해요.
    ).astype(int)

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        df, target, test_size=0.25, random_state=42, stratify=target
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    num_cols = ["study_hours", "sleep_hours"]
    # 설명: 값을 저장하거나 바꿔요.
    cat_cols = ["city"]

    # 설명: 값을 저장하거나 바꿔요.
    preprocessor = ColumnTransformer(
        # 설명: 값을 저장하거나 바꿔요.
        transformers=[
            # 설명: 다음 코드를 실행해요.
            (
                # 설명: 다음 코드를 실행해요.
                "num",
                # 설명: 다음 코드를 실행해요.
                Pipeline(
                    # 설명: 다음 코드를 실행해요.
                    [
                        # 설명: 값을 저장하거나 바꿔요.
                        ("imputer", SimpleImputer(strategy="mean")),
                        # 설명: 다음 코드를 실행해요.
                        ("scaler", StandardScaler()),
                    # 설명: 다음 코드를 실행해요.
                    ]
                # 설명: 다음 코드를 실행해요.
                ),
                # 설명: 다음 코드를 실행해요.
                num_cols,
            # 설명: 다음 코드를 실행해요.
            ),
            # 설명: 다음 코드를 실행해요.
            (
                # 설명: 다음 코드를 실행해요.
                "cat",
                # 설명: 다음 코드를 실행해요.
                Pipeline(
                    # 설명: 다음 코드를 실행해요.
                    [
                        # 설명: 값을 저장하거나 바꿔요.
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # 설명: 값을 저장하거나 바꿔요.
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    # 설명: 다음 코드를 실행해요.
                    ]
                # 설명: 다음 코드를 실행해요.
                ),
                # 설명: 다음 코드를 실행해요.
                cat_cols,
            # 설명: 다음 코드를 실행해요.
            ),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            ("preprocessor", preprocessor),
            # 설명: 값을 저장하거나 바꿔요.
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    acc = float(accuracy_score(y_test, pred))

    # 설명: 값을 저장하거나 바꿔요.
    transformed_sample = model.named_steps["preprocessor"].transform(X_test.head(3))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter44",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "train_rows": int(len(X_train)),
        # 설명: 다음 코드를 실행해요.
        "test_rows": int(len(X_test)),
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(acc, 4),
        # 설명: 다음 코드를 실행해요.
        "transformed_shape_sample": list(transformed_sample.shape),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
