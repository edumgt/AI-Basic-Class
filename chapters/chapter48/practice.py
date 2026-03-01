# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""F1 점수 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import f1_score, precision_score, recall_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "F1 점수는 정밀도와 재현율의 균형을 한 숫자로 본다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "불균형 데이터에서 F1이 높은 모델을 찾는다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([0] * 80 + [1] * 20, dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    pred_high_precision = np.array(([0] * 78) + ([1] * 2) + ([0] * 10) + ([1] * 10), dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    pred_high_recall = np.array(([0] * 65) + ([1] * 15) + ([0] * 3) + ([1] * 17), dtype=int)

    # 설명: `report` 함수를 만들어요.
    def report(pred: np.ndarray) -> dict[str, float]:
        # 설명: 함수 결과를 돌려줘요.
        return {
            # 설명: 값을 저장하거나 바꿔요.
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            # 설명: 값을 저장하거나 바꿔요.
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            # 설명: 값을 저장하거나 바꿔요.
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        # 설명: 다음 코드를 실행해요.
        }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter48",
        # 설명: 다음 코드를 실행해요.
        "topic": "F1 점수",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "model_high_precision": report(pred_high_precision),
        # 설명: 다음 코드를 실행해요.
        "model_high_recall": report(pred_high_recall),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
