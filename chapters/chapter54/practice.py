"""피처 중요도 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


LESSON_10MIN = "트리 기반 모델은 어떤 입력이 중요한지 점수로 보여줄 수 있다."
PRACTICE_30MIN = "랜덤포레스트의 feature importance를 정렬해 본다."


def run() -> dict:
    X, y = make_classification(
        n_samples=260,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    model = RandomForestClassifier(n_estimators=160, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    ranking = sorted(
        [
            {
                "feature": name,
                "importance": round(float(score), 4),
            }
            for name, score in zip(feature_names, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "chapter": "chapter54",
        "topic": "피처 중요도",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "importance_ranking": ranking,
        "importance_sum": round(float(np.sum(importances)), 4),
    }


if __name__ == "__main__":
    print(run())
