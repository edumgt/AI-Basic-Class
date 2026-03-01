"""정밀도와 재현율 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score


LESSON_10MIN = "임계값을 바꾸면 정밀도와 재현율이 서로 반대로 움직일 수 있다."
PRACTICE_30MIN = "여러 threshold에서 precision/recall을 비교한다."


def run() -> dict:
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    y_prob = np.array([0.91, 0.35, 0.76, 0.64, 0.21, 0.55, 0.72, 0.18, 0.43, 0.30])

    thresholds = [0.3, 0.5, 0.7]
    metrics = []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        metrics.append(
            {
                "threshold": th,
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            }
        )

    return {
        "chapter": "chapter47",
        "topic": "정밀도와 재현율",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "threshold_metrics": metrics,
    }


if __name__ == "__main__":
    print(run())
