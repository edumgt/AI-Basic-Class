"""부스팅 심화 개념 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "부스팅은 약한 모델을 순차적으로 보정하며 성능을 높인다."
PRACTICE_30MIN = "learning_rate와 n_estimators 조합을 비교한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    configs = [
        {"learning_rate": 0.03, "n_estimators": 220},
        {"learning_rate": 0.1, "n_estimators": 120},
        {"learning_rate": 0.3, "n_estimators": 60},
    ]

    scores = []
    for cfg in configs:
        model = GradientBoostingClassifier(
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        score = float(accuracy_score(y_test, model.predict(X_test)))
        scores.append(
            {
                "learning_rate": cfg["learning_rate"],
                "n_estimators": cfg["n_estimators"],
                "accuracy": round(score, 4),
            }
        )

    best = max(scores, key=lambda x: x["accuracy"])

    return {
        "chapter": "chapter61",
        "topic": "부스팅 심화 개념",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "results": scores,
        "best_config": best,
    }


if __name__ == "__main__":
    print(run())
