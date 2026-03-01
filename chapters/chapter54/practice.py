# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""피처 중요도 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestClassifier


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "트리 기반 모델은 어떤 입력이 중요한지 점수로 보여줄 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "랜덤포레스트의 feature importance를 정렬해 본다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=260,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=6,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=3,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=1,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # 설명: 값을 저장하거나 바꿔요.
    model = RandomForestClassifier(n_estimators=160, random_state=42)
    # 설명: 다음 코드를 실행해요.
    model.fit(X, y)

    # 설명: 값을 저장하거나 바꿔요.
    importances = model.feature_importances_
    # 설명: 값을 저장하거나 바꿔요.
    ranking = sorted(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "feature": name,
                # 설명: 다음 코드를 실행해요.
                "importance": round(float(score), 4),
            # 설명: 다음 코드를 실행해요.
            }
            # 설명: 같은 동작을 여러 번 반복해요.
            for name, score in zip(feature_names, importances)
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        key=lambda x: x["importance"],
        # 설명: 값을 저장하거나 바꿔요.
        reverse=True,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter54",
        # 설명: 다음 코드를 실행해요.
        "topic": "피처 중요도",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "importance_ranking": ranking,
        # 설명: 다음 코드를 실행해요.
        "importance_sum": round(float(np.sum(importances)), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
