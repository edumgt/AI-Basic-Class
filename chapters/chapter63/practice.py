"""임계값 최적화 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "0.5 고정 대신 목표에 맞는 threshold를 찾으면 성능 균형을 맞출 수 있다."
PRACTICE_30MIN = "여러 임계값에서 precision/recall/F1을 계산해 최적값을 고른다."


def run() -> dict:
    X, y = make_classification(
        n_samples=420,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.88, 0.12],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    threshold_scores = []
    for th in np.linspace(0.2, 0.8, 7):
        pred = (prob >= th).astype(int)
        threshold_scores.append(
            {
                "threshold": round(float(th), 2),
                "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            }
        )

    best = max(threshold_scores, key=lambda x: x["f1"])

    return {
        "chapter": "chapter63",
        "topic": "임계값 최적화",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "threshold_scores": threshold_scores,
        "best_threshold_by_f1": best,
    }


if __name__ == "__main__":
    print(run())
