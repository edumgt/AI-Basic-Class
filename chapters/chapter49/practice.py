"""ROC-AUC 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve


LESSON_10MIN = "ROC 곡선은 임계값을 바꿔도 모델이 얼마나 잘 구분하는지 보여준다."
PRACTICE_30MIN = "ROC 곡선 좌표와 AUC 값을 계산한다."


def run() -> dict:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    y_score = np.array([0.12, 0.33, 0.88, 0.74, 0.41, 0.92, 0.25, 0.61, 0.80, 0.30])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = float(auc(fpr, tpr))

    points = [
        {
            "fpr": round(float(fpr[i]), 4),
            "tpr": round(float(tpr[i]), 4),
            "threshold": round(float(thresholds[i]), 4),
        }
        for i in range(len(fpr))
    ]

    return {
        "chapter": "chapter49",
        "topic": "ROC-AUC",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "auc": round(auc_value, 4),
        "roc_points": points,
    }


if __name__ == "__main__":
    print(run())
