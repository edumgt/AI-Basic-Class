"""최종 미니 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LESSON_10MIN = "최종 프로젝트는 문제정의-전처리-학습-평가-보고까지 한 흐름으로 완성한다."
PRACTICE_30MIN = "합성 데이터 하나로 end-to-end 분류 파이프라인을 실행한다."


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 260

    df = pd.DataFrame(
        {
            "study_minutes": rng.normal(45, 15, size=n),
            "attendance_rate": rng.uniform(0.6, 1.0, size=n),
            "quiz_score": rng.normal(70, 12, size=n),
        }
    )

    y = (
        0.02 * df["study_minutes"]
        + 1.8 * df["attendance_rate"]
        + 0.03 * df["quiz_score"]
        + rng.normal(0, 0.35, size=n)
        > 4.2
    ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "chapter": "chapter98",
        "topic": "최종 미니 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "pipeline_steps": ["scaler", "classifier"],
    }


if __name__ == "__main__":
    print(run())
