"""하이퍼파라미터 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


LESSON_10MIN = "하이퍼파라미터는 사람이 정하는 설정값이며 성능에 큰 영향을 준다."
PRACTICE_30MIN = "max_depth를 바꿔 성능 변화를 관찰한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=280,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    depth_options = [2, 4, 6, None]
    results: dict[str, float] = {}
    best_depth: str | None = None
    best_score = -1.0

    for depth in depth_options:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        score = float(accuracy_score(y_test, model.predict(X_test)))
        key = "None" if depth is None else str(depth)
        results[key] = round(score, 4)
        if score > best_score:
            best_score = score
            best_depth = key

    return {
        "chapter": "chapter56",
        "topic": "하이퍼파라미터",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "accuracy_by_max_depth": results,
        "best_max_depth": best_depth,
    }


if __name__ == "__main__":
    print(run())
