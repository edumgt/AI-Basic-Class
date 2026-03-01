"""배깅과 부스팅 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


LESSON_10MIN = "배깅은 병렬 평균, 부스팅은 순차 보정이라는 차이가 있다."
PRACTICE_30MIN = "단일 트리, 배깅, 부스팅 성능을 비교한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=340,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        n_estimators=80,
        random_state=42,
    )
    boosting = GradientBoostingClassifier(random_state=42)

    models = {
        "single_tree": tree,
        "bagging": bagging,
        "boosting": boosting,
    }

    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = round(float(accuracy_score(y_test, pred)), 4)

    return {
        "chapter": "chapter60",
        "topic": "배깅과 부스팅",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "accuracy": scores,
    }


if __name__ == "__main__":
    print(run())
