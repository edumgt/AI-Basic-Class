"""정확도 함정 이해 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


LESSON_10MIN = "데이터가 한쪽으로 치우치면 정확도만으로는 모델 품질을 판단하기 어렵다."
PRACTICE_30MIN = "같은 문제에서 두 모델의 정확도와 F1을 비교한다."


def _metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run() -> dict:
    y_true = np.array([0] * 95 + [1] * 5, dtype=int)

    model_a_pred = np.array([0] * 100, dtype=int)
    model_b_pred = np.array(([0] * 90) + ([1] * 10), dtype=int)

    return {
        "chapter": "chapter45",
        "topic": "정확도 함정 이해",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "positive_ratio": round(float(y_true.mean()), 4),
        "model_a_all_zero": _metric_set(y_true, model_a_pred),
        "model_b_predict_some_positive": _metric_set(y_true, model_b_pred),
    }


if __name__ == "__main__":
    print(run())
