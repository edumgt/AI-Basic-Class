"""F1 점수 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


LESSON_10MIN = "F1 점수는 정밀도와 재현율의 균형을 한 숫자로 본다."
PRACTICE_30MIN = "불균형 데이터에서 F1이 높은 모델을 찾는다."


def run() -> dict:
    y_true = np.array([0] * 80 + [1] * 20, dtype=int)

    pred_high_precision = np.array(([0] * 78) + ([1] * 2) + ([0] * 10) + ([1] * 10), dtype=int)
    pred_high_recall = np.array(([0] * 65) + ([1] * 15) + ([0] * 3) + ([1] * 17), dtype=int)

    def report(pred: np.ndarray) -> dict[str, float]:
        return {
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        }

    return {
        "chapter": "chapter48",
        "topic": "F1 점수",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "model_high_precision": report(pred_high_precision),
        "model_high_recall": report(pred_high_recall),
    }


if __name__ == "__main__":
    print(run())
