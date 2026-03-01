# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""클래스 불균형 처리 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import f1_score, precision_score, recall_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "불균형 데이터에서는 class_weight 등 보정 기법이 필요할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "기본 모델과 class_weight=balanced 모델을 비교한다."


# 설명: `_scores` 함수를 만들어요.
def _scores(y_true, y_pred) -> dict[str, float]:
    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 값을 저장하거나 바꿔요.
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=500,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=12,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=6,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=2,
        # 설명: 값을 저장하거나 바꿔요.
        weights=[0.93, 0.07],
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    base = LogisticRegression(max_iter=500, random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    balanced = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)

    # 설명: 다음 코드를 실행해요.
    base.fit(X_train, y_train)
    # 설명: 다음 코드를 실행해요.
    balanced.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred_base = base.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    pred_balanced = balanced.predict(X_test)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter62",
        # 설명: 다음 코드를 실행해요.
        "topic": "클래스 불균형 처리",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "base_model": _scores(y_test, pred_base),
        # 설명: 다음 코드를 실행해요.
        "balanced_model": _scores(y_test, pred_balanced),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
