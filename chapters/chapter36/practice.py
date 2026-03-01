"""스케일링 개념 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LESSON_10MIN = "서로 단위가 다른 특성은 스케일을 맞추면 학습이 안정적이다."
PRACTICE_30MIN = "스케일링 전후 분류 정확도를 비교한다."


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 180

    feature_large = rng.normal(loc=5000, scale=900, size=n)
    feature_small = rng.normal(loc=0.5, scale=0.12, size=n)
    noise = rng.normal(loc=0.0, scale=0.06, size=n)
    y = (feature_small + noise > 0.5).astype(int)

    X = np.column_stack([feature_large, feature_small])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    raw_model = LogisticRegression(max_iter=500)
    raw_model.fit(X_train, y_train)
    raw_pred = raw_model.predict(X_test)
    raw_acc = float(accuracy_score(y_test, raw_pred))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaled_model = LogisticRegression(max_iter=500)
    scaled_model.fit(X_train_scaled, y_train)
    scaled_pred = scaled_model.predict(X_test_scaled)
    scaled_acc = float(accuracy_score(y_test, scaled_pred))

    return {
        "chapter": "chapter36",
        "topic": "스케일링 개념",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "raw_feature_mean": X.mean(axis=0).round(3).tolist(),
        "raw_feature_std": X.std(axis=0).round(3).tolist(),
        "scaled_train_mean": X_train_scaled.mean(axis=0).round(3).tolist(),
        "scaled_train_std": X_train_scaled.std(axis=0).round(3).tolist(),
        "accuracy_raw": round(raw_acc, 4),
        "accuracy_scaled": round(scaled_acc, 4),
    }


if __name__ == "__main__":
    print(run())
