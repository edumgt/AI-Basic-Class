"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


LESSON_10MIN = "문제 성격에 맞는 지표를 먼저 정하고 모델을 평가해야 한다."
PRACTICE_30MIN = "불균형 문제에서 어떤 지표를 선택할지 근거를 정리한다."


def run() -> dict:
    # 가상의 사기 탐지 문제: 1이 사기(소수 클래스)
    y_true = np.array([0] * 180 + [1] * 20, dtype=int)

    pred_model = np.array(([0] * 168) + ([1] * 12) + ([0] * 7) + ([1] * 13), dtype=int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, pred_model)), 4),
        "precision": round(float(precision_score(y_true, pred_model, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred_model, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred_model, zero_division=0)), 4),
    }

    selected_metric = "recall"
    reason = "사기(양성)를 놓치면 비용이 크기 때문에 재현율을 우선으로 본다."

    return {
        "chapter": "chapter55",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "metrics": metrics,
        "selected_metric": selected_metric,
        "selection_reason": reason,
    }


if __name__ == "__main__":
    print(run())
