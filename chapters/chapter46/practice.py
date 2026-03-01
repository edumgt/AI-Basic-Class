# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""혼동행렬 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import confusion_matrix


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "혼동행렬은 정답과 예측의 조합을 한 번에 보여준다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "TN, FP, FN, TP 값을 직접 읽어본다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    # 설명: 값을 저장하거나 바꿔요.
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0], dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    cm = confusion_matrix(y_true, y_pred)
    # 설명: 값을 저장하거나 바꿔요.
    tn, fp, fn, tp = cm.ravel()

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter46",
        # 설명: 다음 코드를 실행해요.
        "topic": "혼동행렬",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "confusion_matrix": cm.tolist(),
        # 설명: 다음 코드를 실행해요.
        "tn": int(tn),
        # 설명: 다음 코드를 실행해요.
        "fp": int(fp),
        # 설명: 다음 코드를 실행해요.
        "fn": int(fn),
        # 설명: 다음 코드를 실행해요.
        "tp": int(tp),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
