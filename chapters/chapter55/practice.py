# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "문제 성격에 맞는 지표를 먼저 정하고 모델을 평가해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "불균형 문제에서 어떤 지표를 선택할지 근거를 정리한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 가상의 사기 탐지 문제: 1이 사기(소수 클래스)
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([0] * 180 + [1] * 20, dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    pred_model = np.array(([0] * 168) + ([1] * 12) + ([0] * 7) + ([1] * 13), dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    metrics = {
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(float(accuracy_score(y_true, pred_model)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "precision": round(float(precision_score(y_true, pred_model, zero_division=0)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "recall": round(float(recall_score(y_true, pred_model, zero_division=0)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "f1": round(float(f1_score(y_true, pred_model, zero_division=0)), 4),
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    selected_metric = "recall"
    # 설명: 값을 저장하거나 바꿔요.
    reason = "사기(양성)를 놓치면 비용이 크기 때문에 재현율을 우선으로 본다."

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter55",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "metrics": metrics,
        # 설명: 다음 코드를 실행해요.
        "selected_metric": selected_metric,
        # 설명: 다음 코드를 실행해요.
        "selection_reason": reason,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
