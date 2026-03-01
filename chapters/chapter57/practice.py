# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""Grid Search 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import GridSearchCV, train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "Grid Search는 정해 둔 후보를 전부 시도해 최적값을 찾는다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "C와 penalty 조합을 탐색해 최적 분류기를 찾는다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=320,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=12,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=7,
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
    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    param_grid = {
        # 설명: 다음 코드를 실행해요.
        "C": [0.1, 1.0, 3.0, 10.0],
        # 설명: 다음 코드를 실행해요.
        "penalty": ["l1", "l2"],
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    search = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="accuracy", n_jobs=None)
    # 설명: 다음 코드를 실행해요.
    search.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    test_score = float(search.score(X_test, y_test))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter57",
        # 설명: 다음 코드를 실행해요.
        "topic": "Grid Search",
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
