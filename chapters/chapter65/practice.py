"""모델 카드 작성 실습 파일"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "모델 카드로 입력, 출력, 제한사항을 문서화하면 운영 리스크를 줄일 수 있다."
PRACTICE_30MIN = "간단한 모델을 학습하고 카드 템플릿을 채운다."


def run() -> dict:
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    card = {
        "model_name": "baseline_logistic_classifier",
        "purpose": "학습자 합격 여부 예측 데모",
        "inputs": ["feature_0", "feature_1", "...", "feature_7"],
        "output": "0 또는 1",
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
        },
        "limitations": [
            "소규모 합성 데이터 기반 데모라 실제 환경 일반화가 제한됨",
            "민감 정보 편향 검토가 별도로 필요함",
        ],
        "monitoring": [
            "월별 정확도 추세",
            "클래스 불균형 변화",
        ],
    }

    return {
        "chapter": "chapter65",
        "topic": "모델 카드 작성",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "model_card": card,
    }


if __name__ == "__main__":
    print(run())
