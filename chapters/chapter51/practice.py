# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""교차검증 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import StratifiedKFold, cross_val_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "교차검증은 데이터를 여러 번 나눠 더 안정적으로 성능을 본다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "5-fold 점수의 평균과 분산을 확인한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=300,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=10,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=6,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=2,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = LogisticRegression(max_iter=500, random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter51",
        # 설명: 다음 코드를 실행해요.
        "topic": "교차검증",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "fold_scores": np.round(scores, 4).tolist(),
        # 설명: 다음 코드를 실행해요.
        "cv_mean": round(float(scores.mean()), 4),
        # 설명: 다음 코드를 실행해요.
        "cv_std": round(float(scores.std()), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
