"""혼동행렬 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


LESSON_10MIN = "혼동행렬은 정답과 예측의 조합을 한 번에 보여준다."
PRACTICE_30MIN = "TN, FP, FN, TP 값을 직접 읽어본다."


def run() -> dict:
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0], dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "chapter": "chapter46",
        "topic": "혼동행렬",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


if __name__ == "__main__":
    print(run())
