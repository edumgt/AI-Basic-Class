"""Random Search 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


LESSON_10MIN = "Random Search는 넓은 후보 공간을 빠르게 탐색할 때 유용하다."
PRACTICE_30MIN = "랜덤포레스트 설정을 랜덤 탐색으로 찾는다."


def run() -> dict:
    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base_model = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [80, 120, 180, 240],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 4, 6],
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=8,
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=None,
    )
    search.fit(X_train, y_train)

    test_score = float(search.score(X_test, y_test))

    return {
        "chapter": "chapter58",
        "topic": "Random Search",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "best_params": search.best_params_,
        "best_cv_score": round(float(search.best_score_), 4),
        "test_accuracy": round(test_score, 4),
    }


if __name__ == "__main__":
    print(run())
