# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""ROC-AUC 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import auc, roc_curve


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "ROC 곡선은 임계값을 바꿔도 모델이 얼마나 잘 구분하는지 보여준다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "ROC 곡선 좌표와 AUC 값을 계산한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    y_score = np.array([0.12, 0.33, 0.88, 0.74, 0.41, 0.92, 0.25, 0.61, 0.80, 0.30])

    # 설명: 값을 저장하거나 바꿔요.
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # 설명: 값을 저장하거나 바꿔요.
    auc_value = float(auc(fpr, tpr))

    # 설명: 값을 저장하거나 바꿔요.
    points = [
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "fpr": round(float(fpr[i]), 4),
            # 설명: 다음 코드를 실행해요.
            "tpr": round(float(tpr[i]), 4),
            # 설명: 다음 코드를 실행해요.
            "threshold": round(float(thresholds[i]), 4),
        # 설명: 다음 코드를 실행해요.
        }
        # 설명: 같은 동작을 여러 번 반복해요.
        for i in range(len(fpr))
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter49",
        # 설명: 다음 코드를 실행해요.
        "topic": "ROC-AUC",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "auc": round(auc_value, 4),
        # 설명: 다음 코드를 실행해요.
        "roc_points": points,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
