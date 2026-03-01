"""앙상블 개념 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


LESSON_10MIN = "여러 모델을 결합하면 단일 모델보다 안정적인 예측을 얻을 수 있다."
PRACTICE_30MIN = "개별 모델과 Voting 앙상블 성능을 비교한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=320,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=500, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=140, random_state=42)

    voting = VotingClassifier(
        estimators=[("lr", lr), ("dt", dt), ("rf", rf)],
        voting="hard",
    )

    models = {
        "logistic_regression": lr,
        "decision_tree": dt,
        "random_forest": rf,
        "voting_ensemble": voting,
    }

    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = round(float(accuracy_score(y_test, pred)), 4)

    return {
        "chapter": "chapter59",
        "topic": "앙상블 개념",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "accuracy": scores,
    }


if __name__ == "__main__":
    print(run())
