# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""재현 가능한 실험 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "random_state를 고정하면 같은 코드에서 같은 결과를 얻기 쉽다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "시드 고정 전/후 예측 결과 차이를 비교한다."


# 설명: `_run_model` 함수를 만들어요.
def _run_model(random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
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
        random_state=123,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = RandomForestClassifier(n_estimators=120, random_state=random_state)
    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test)
    # 설명: 함수 결과를 돌려줘요.
    return pred, y_test


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    pred_free_1, y_test = _run_model(random_state=None)
    # 설명: 값을 저장하거나 바꿔요.
    pred_free_2, _ = _run_model(random_state=None)

    # 설명: 값을 저장하거나 바꿔요.
    pred_fixed_1, y_test_fixed = _run_model(random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    pred_fixed_2, _ = _run_model(random_state=42)

    # 설명: 값을 저장하거나 바꿔요.
    free_diff = int((pred_free_1 != pred_free_2).sum())
    # 설명: 값을 저장하거나 바꿔요.
    fixed_diff = int((pred_fixed_1 != pred_fixed_2).sum())

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter43",
        # 설명: 다음 코드를 실행해요.
        "topic": "재현 가능한 실험",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "accuracy_fixed": round(float(accuracy_score(y_test_fixed, pred_fixed_1)), 4),
        # 설명: 다음 코드를 실행해요.
        "prediction_diff_without_seed": free_diff,
        # 설명: 다음 코드를 실행해요.
        "prediction_diff_with_seed": fixed_diff,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
