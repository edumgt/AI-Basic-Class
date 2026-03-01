"""Grid Search 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split


LESSON_10MIN = "Grid Search는 정해 둔 후보를 전부 시도해 최적값을 찾는다."
PRACTICE_30MIN = "C와 penalty 조합을 탐색해 최적 분류기를 찾는다."


def run() -> dict:
    X, y = make_classification(
        n_samples=320,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    param_grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        "penalty": ["l1", "l2"],
    }

    search = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="accuracy", n_jobs=None)
    search.fit(X_train, y_train)

    test_score = float(search.score(X_test, y_test))

    return {
        "chapter": "chapter57",
        "topic": "Grid Search",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "best_params": search.best_params_,
        "best_cv_score": round(float(search.best_score_), 4),
        "test_accuracy": round(test_score, 4),
    }


if __name__ == "__main__":
    print(run())
