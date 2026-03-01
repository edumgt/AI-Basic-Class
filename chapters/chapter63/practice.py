# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""임계값 최적화 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import f1_score, precision_score, recall_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "0.5 고정 대신 목표에 맞는 threshold를 찾으면 성능 균형을 맞출 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "여러 임계값에서 precision/recall/F1을 계산해 최적값을 고른다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=420,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=10,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=6,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=2,
        # 설명: 값을 저장하거나 바꿔요.
        weights=[0.88, 0.12],
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    prob = model.predict_proba(X_test)[:, 1]

    # 설명: 값을 저장하거나 바꿔요.
    threshold_scores = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for th in np.linspace(0.2, 0.8, 7):
        # 설명: 값을 저장하거나 바꿔요.
        pred = (prob >= th).astype(int)
        # 설명: 다음 코드를 실행해요.
        threshold_scores.append(
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "threshold": round(float(th), 2),
                # 설명: 값을 저장하거나 바꿔요.
                "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
                # 설명: 값을 저장하거나 바꿔요.
                "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                # 설명: 값을 저장하거나 바꿔요.
                "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            # 설명: 다음 코드를 실행해요.
            }
        # 설명: 다음 코드를 실행해요.
        )

    # 설명: 값을 저장하거나 바꿔요.
    best = max(threshold_scores, key=lambda x: x["f1"])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter63",
        # 설명: 다음 코드를 실행해요.
        "topic": "임계값 최적화",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "threshold_scores": threshold_scores,
        # 설명: 다음 코드를 실행해요.
        "best_threshold_by_f1": best,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
