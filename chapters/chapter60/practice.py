# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""배깅과 부스팅 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.tree import DecisionTreeClassifier


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "배깅은 병렬 평균, 부스팅은 순차 보정이라는 차이가 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "단일 트리, 배깅, 부스팅 성능을 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=340,
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
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    bagging = BaggingClassifier(
        # 설명: 값을 저장하거나 바꿔요.
        estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        # 설명: 값을 저장하거나 바꿔요.
        n_estimators=80,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    boosting = GradientBoostingClassifier(random_state=42)

    # 설명: 값을 저장하거나 바꿔요.
    models = {
        # 설명: 다음 코드를 실행해요.
        "single_tree": tree,
        # 설명: 다음 코드를 실행해요.
        "bagging": bagging,
        # 설명: 다음 코드를 실행해요.
        "boosting": boosting,
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    scores: dict[str, float] = {}
    # 설명: 같은 동작을 여러 번 반복해요.
    for name, model in models.items():
        # 설명: 다음 코드를 실행해요.
        model.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        scores[name] = round(float(accuracy_score(y_test, pred)), 4)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter60",
        # 설명: 다음 코드를 실행해요.
        "topic": "배깅과 부스팅",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "accuracy": scores,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
