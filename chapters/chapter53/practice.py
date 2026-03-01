"""학습곡선 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve


LESSON_10MIN = "학습곡선으로 데이터 양이 늘 때 성능이 어떻게 변하는지 볼 수 있다."
PRACTICE_30MIN = "train/test score 곡선을 계산해 추세를 확인한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=320,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=500, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )

    return {
        "chapter": "chapter53",
        "topic": "학습곡선",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "train_sizes": train_sizes.tolist(),
        "train_score_mean": np.round(train_scores.mean(axis=1), 4).tolist(),
        "valid_score_mean": np.round(valid_scores.mean(axis=1), 4).tolist(),
    }


if __name__ == "__main__":
    print(run())
