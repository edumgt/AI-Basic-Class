# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""모델 카드 작성 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score, f1_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "모델 카드로 입력, 출력, 제한사항을 문서화하면 운영 리스크를 줄일 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "간단한 모델을 학습하고 카드 템플릿을 채운다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=260,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=8,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=5,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=1,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.25, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = LogisticRegression(max_iter=500, random_state=42)
    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    card = {
        # 설명: 다음 코드를 실행해요.
        "model_name": "baseline_logistic_classifier",
        # 설명: 다음 코드를 실행해요.
        "purpose": "학습자 합격 여부 예측 데모",
        # 설명: 다음 코드를 실행해요.
        "inputs": ["feature_0", "feature_1", "...", "feature_7"],
        # 설명: 다음 코드를 실행해요.
        "output": "0 또는 1",
        # 설명: 다음 코드를 실행해요.
        "metrics": {
            # 설명: 다음 코드를 실행해요.
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            # 설명: 다음 코드를 실행해요.
            "f1": round(float(f1_score(y_test, pred)), 4),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "limitations": [
            # 설명: 다음 코드를 실행해요.
            "소규모 합성 데이터 기반 데모라 실제 환경 일반화가 제한됨",
            # 설명: 다음 코드를 실행해요.
            "민감 정보 편향 검토가 별도로 필요함",
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 다음 코드를 실행해요.
        "monitoring": [
            # 설명: 다음 코드를 실행해요.
            "월별 정확도 추세",
            # 설명: 다음 코드를 실행해요.
            "클래스 불균형 변화",
        # 설명: 다음 코드를 실행해요.
        ],
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter65",
        # 설명: 다음 코드를 실행해요.
        "topic": "모델 카드 작성",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "model_card": card,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
