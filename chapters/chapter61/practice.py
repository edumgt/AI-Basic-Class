# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""부스팅 심화 개념 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import GradientBoostingClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "부스팅은 약한 모델을 순차적으로 보정하며 성능을 높인다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "learning_rate와 n_estimators 조합을 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=360,
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
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    configs = [
        # 설명: 다음 코드를 실행해요.
        {"learning_rate": 0.03, "n_estimators": 220},
        # 설명: 다음 코드를 실행해요.
        {"learning_rate": 0.1, "n_estimators": 120},
        # 설명: 다음 코드를 실행해요.
        {"learning_rate": 0.3, "n_estimators": 60},
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 값을 저장하거나 바꿔요.
    scores = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for cfg in configs:
        # 설명: 값을 저장하거나 바꿔요.
        model = GradientBoostingClassifier(
            # 설명: 값을 저장하거나 바꿔요.
            learning_rate=cfg["learning_rate"],
            # 설명: 값을 저장하거나 바꿔요.
            n_estimators=cfg["n_estimators"],
            # 설명: 값을 저장하거나 바꿔요.
            random_state=42,
        # 설명: 다음 코드를 실행해요.
        )
        # 설명: 다음 코드를 실행해요.
        model.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        score = float(accuracy_score(y_test, model.predict(X_test)))
        # 설명: 다음 코드를 실행해요.
        scores.append(
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "learning_rate": cfg["learning_rate"],
                # 설명: 다음 코드를 실행해요.
                "n_estimators": cfg["n_estimators"],
                # 설명: 다음 코드를 실행해요.
                "accuracy": round(score, 4),
            # 설명: 다음 코드를 실행해요.
            }
        # 설명: 다음 코드를 실행해요.
        )

    # 설명: 값을 저장하거나 바꿔요.
    best = max(scores, key=lambda x: x["accuracy"])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter61",
        # 설명: 다음 코드를 실행해요.
        "topic": "부스팅 심화 개념",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "results": scores,
        # 설명: 다음 코드를 실행해요.
        "best_config": best,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
