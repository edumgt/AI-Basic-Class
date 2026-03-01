"""교차검증 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


LESSON_10MIN = "교차검증은 데이터를 여러 번 나눠 더 안정적으로 성능을 본다."
PRACTICE_30MIN = "5-fold 점수의 평균과 분산을 확인한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=500, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return {
        "chapter": "chapter51",
        "topic": "교차검증",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "fold_scores": np.round(scores, 4).tolist(),
        "cv_mean": round(float(scores.mean()), 4),
        "cv_std": round(float(scores.std()), 4),
    }


if __name__ == "__main__":
    print(run())
