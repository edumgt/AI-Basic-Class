# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""학습곡선 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import StratifiedKFold, learning_curve


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "학습곡선으로 데이터 양이 늘 때 성능이 어떻게 변하는지 볼 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "train/test score 곡선을 계산해 추세를 확인한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=320,
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
    train_sizes, train_scores, valid_scores = learning_curve(
        # 설명: 다음 코드를 실행해요.
        model,
        # 설명: 다음 코드를 실행해요.
        X,
        # 설명: 다음 코드를 실행해요.
        y,
        # 설명: 값을 저장하거나 바꿔요.
        train_sizes=np.linspace(0.2, 1.0, 5),
        # 설명: 값을 저장하거나 바꿔요.
        cv=cv,
        # 설명: 값을 저장하거나 바꿔요.
        scoring="accuracy",
        # 설명: 값을 저장하거나 바꿔요.
        n_jobs=None,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter53",
        # 설명: 다음 코드를 실행해요.
        "topic": "학습곡선",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "train_sizes": train_sizes.tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "train_score_mean": np.round(train_scores.mean(axis=1), 4).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "valid_score_mean": np.round(valid_scores.mean(axis=1), 4).tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
