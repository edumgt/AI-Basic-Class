"""재현 가능한 실험 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "random_state를 고정하면 같은 코드에서 같은 결과를 얻기 쉽다."
PRACTICE_30MIN = "시드 고정 전/후 예측 결과 차이를 비교한다."


def _run_model(random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=120, random_state=random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, y_test


def run() -> dict:
    pred_free_1, y_test = _run_model(random_state=None)
    pred_free_2, _ = _run_model(random_state=None)

    pred_fixed_1, y_test_fixed = _run_model(random_state=42)
    pred_fixed_2, _ = _run_model(random_state=42)

    free_diff = int((pred_free_1 != pred_free_2).sum())
    fixed_diff = int((pred_fixed_1 != pred_fixed_2).sum())

    return {
        "chapter": "chapter43",
        "topic": "재현 가능한 실험",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "accuracy_fixed": round(float(accuracy_score(y_test_fixed, pred_fixed_1)), 4),
        "prediction_diff_without_seed": free_diff,
        "prediction_diff_with_seed": fixed_diff,
    }


if __name__ == "__main__":
    print(run())
