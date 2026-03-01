# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""정밀도와 재현율 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import precision_score, recall_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "임계값을 바꾸면 정밀도와 재현율이 서로 반대로 움직일 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "여러 threshold에서 precision/recall을 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    y_prob = np.array([0.91, 0.35, 0.76, 0.64, 0.21, 0.55, 0.72, 0.18, 0.43, 0.30])

    # 설명: 값을 저장하거나 바꿔요.
    thresholds = [0.3, 0.5, 0.7]
    # 설명: 값을 저장하거나 바꿔요.
    metrics = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for th in thresholds:
        # 설명: 값을 저장하거나 바꿔요.
        pred = (y_prob >= th).astype(int)
        # 설명: 다음 코드를 실행해요.
        metrics.append(
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "threshold": th,
                # 설명: 값을 저장하거나 바꿔요.
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                # 설명: 값을 저장하거나 바꿔요.
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            # 설명: 다음 코드를 실행해요.
            }
        # 설명: 다음 코드를 실행해요.
        )

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter47",
        # 설명: 다음 코드를 실행해요.
        "topic": "정밀도와 재현율",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "threshold_metrics": metrics,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
