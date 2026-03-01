# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""정확도 함정 이해 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "데이터가 한쪽으로 치우치면 정확도만으로는 모델 품질을 판단하기 어렵다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "같은 문제에서 두 모델의 정확도와 F1을 비교한다."


# 설명: `_metric_set` 함수를 만들어요.
def _metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
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
    y_true = np.array([0] * 95 + [1] * 5, dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    model_a_pred = np.array([0] * 100, dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    model_b_pred = np.array(([0] * 90) + ([1] * 10), dtype=int)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter45",
        # 설명: 다음 코드를 실행해요.
        "topic": "정확도 함정 이해",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "positive_ratio": round(float(y_true.mean()), 4),
        # 설명: 다음 코드를 실행해요.
        "model_a_all_zero": _metric_set(y_true, model_a_pred),
        # 설명: 다음 코드를 실행해요.
        "model_b_predict_some_positive": _metric_set(y_true, model_b_pred),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
