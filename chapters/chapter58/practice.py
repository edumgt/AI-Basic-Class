# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""Random Search 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "Random Search는 넓은 후보 공간을 빠르게 탐색할 때 유용하다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "랜덤포레스트 설정을 랜덤 탐색으로 찾는다."


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
        X, y, test_size=0.25, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    base_model = RandomForestClassifier(random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    param_dist = {
        # 설명: 다음 코드를 실행해요.
        "n_estimators": [80, 120, 180, 240],
        # 설명: 다음 코드를 실행해요.
        "max_depth": [3, 5, 8, None],
        # 설명: 다음 코드를 실행해요.
        "min_samples_split": [2, 4, 6],
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    search = RandomizedSearchCV(
        # 설명: 다음 코드를 실행해요.
        base_model,
        # 설명: 값을 저장하거나 바꿔요.
        param_distributions=param_dist,
        # 설명: 값을 저장하거나 바꿔요.
        n_iter=8,
        # 설명: 값을 저장하거나 바꿔요.
        cv=4,
        # 설명: 값을 저장하거나 바꿔요.
        scoring="accuracy",
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
        # 설명: 값을 저장하거나 바꿔요.
        n_jobs=None,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 다음 코드를 실행해요.
    search.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    test_score = float(search.score(X_test, y_test))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter58",
        # 설명: 다음 코드를 실행해요.
        "topic": "Random Search",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "best_params": search.best_params_,
        # 설명: 다음 코드를 실행해요.
        "best_cv_score": round(float(search.best_score_), 4),
        # 설명: 다음 코드를 실행해요.
        "test_accuracy": round(test_score, 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
