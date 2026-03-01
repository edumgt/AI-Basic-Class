"""클래스 불균형 처리 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "불균형 데이터에서는 class_weight 등 보정 기법이 필요할 수 있다."
PRACTICE_30MIN = "기본 모델과 class_weight=balanced 모델을 비교한다."


def _scores(y_true, y_pred) -> dict[str, float]:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run() -> dict:
    X, y = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.93, 0.07],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    base = LogisticRegression(max_iter=500, random_state=42)
    balanced = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)

    base.fit(X_train, y_train)
    balanced.fit(X_train, y_train)

    pred_base = base.predict(X_test)
    pred_balanced = balanced.predict(X_test)

    return {
        "chapter": "chapter62",
        "topic": "클래스 불균형 처리",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "base_model": _scores(y_test, pred_base),
        "balanced_model": _scores(y_test, pred_balanced),
    }


if __name__ == "__main__":
    print(run())
