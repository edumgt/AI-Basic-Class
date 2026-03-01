"""파이프라인 기초 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LESSON_10MIN = "전처리와 모델을 파이프라인으로 묶으면 실수를 줄일 수 있다."
PRACTICE_30MIN = "StandardScaler와 LogisticRegression을 한 줄 흐름으로 실행한다."


def run() -> dict:
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    return {
        "chapter": "chapter37",
        "topic": "파이프라인 기초",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "pipeline_steps": list(pipe.named_steps.keys()),
        "accuracy": round(acc, 4),
        "sample_predictions": pred[:10].tolist(),
    }


if __name__ == "__main__":
    print(run())
