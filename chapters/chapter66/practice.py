"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


LESSON_10MIN = "기준 모델과 개선 모델을 같은 테스트셋에서 비교해야 개선 폭을 신뢰할 수 있다."
PRACTICE_30MIN = "baseline 대비 개선률을 계산해 보고서 형태로 출력한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=420,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    baseline = DecisionTreeClassifier(max_depth=3, random_state=42)
    improved = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)

    f1_base = float(f1_score(y_test, baseline.predict(X_test)))
    f1_improved = float(f1_score(y_test, improved.predict(X_test)))

    improvement_pct = float((f1_improved - f1_base) / max(abs(f1_base), 1e-9) * 100)

    return {
        "chapter": "chapter66",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "baseline_f1": round(f1_base, 4),
        "improved_f1": round(f1_improved, 4),
        "improvement_percent": round(improvement_pct, 2),
    }


if __name__ == "__main__":
    print(run())
