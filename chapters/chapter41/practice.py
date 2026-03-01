"""간단한 피처 엔지니어링 실습 파일"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "원본 특성만 쓰는 것보다 새 특성을 만들면 성능이 좋아질 수 있다."
PRACTICE_30MIN = "파생 특성 추가 전/후 R2를 비교한다."


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 200

    study_hours = rng.uniform(1, 6, size=n)
    sleep_hours = rng.uniform(5, 9, size=n)

    score = 8 * study_hours + 4 * sleep_hours + 2 * study_hours * sleep_hours + rng.normal(0, 2.0, size=n)

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "score": score,
        }
    )

    base_features = ["study_hours", "sleep_hours"]
    engineered = df.copy()
    engineered["study_x_sleep"] = engineered["study_hours"] * engineered["sleep_hours"]
    engineered["study_per_sleep"] = engineered["study_hours"] / engineered["sleep_hours"]

    X_base = df[base_features]
    X_eng = engineered[["study_hours", "sleep_hours", "study_x_sleep", "study_per_sleep"]]
    y = df["score"]

    Xb_train, Xb_test, y_train, y_test = train_test_split(
        X_base, y, test_size=0.25, random_state=42
    )
    Xe_train, Xe_test, _, _ = train_test_split(X_eng, y, test_size=0.25, random_state=42)

    base_model = LinearRegression().fit(Xb_train, y_train)
    eng_model = LinearRegression().fit(Xe_train, y_train)

    base_r2 = float(r2_score(y_test, base_model.predict(Xb_test)))
    eng_r2 = float(r2_score(y_test, eng_model.predict(Xe_test)))

    return {
        "chapter": "chapter41",
        "topic": "간단한 피처 엔지니어링",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "base_features": base_features,
        "engineered_features": ["study_x_sleep", "study_per_sleep"],
        "r2_base": round(base_r2, 4),
        "r2_engineered": round(eng_r2, 4),
    }


if __name__ == "__main__":
    print(run())
