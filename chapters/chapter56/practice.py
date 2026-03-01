# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""하이퍼파라미터 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.tree import DecisionTreeClassifier


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "하이퍼파라미터는 사람이 정하는 설정값이며 성능에 큰 영향을 준다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "max_depth를 바꿔 성능 변화를 관찰한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=280,
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
    depth_options = [2, 4, 6, None]
    # 설명: 값을 저장하거나 바꿔요.
    results: dict[str, float] = {}
    # 설명: 값을 저장하거나 바꿔요.
    best_depth: str | None = None
    # 설명: 값을 저장하거나 바꿔요.
    best_score = -1.0

    # 설명: 같은 동작을 여러 번 반복해요.
    for depth in depth_options:
        # 설명: 값을 저장하거나 바꿔요.
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        # 설명: 다음 코드를 실행해요.
        model.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        score = float(accuracy_score(y_test, model.predict(X_test)))
        # 설명: 값을 저장하거나 바꿔요.
        key = "None" if depth is None else str(depth)
        # 설명: 값을 저장하거나 바꿔요.
        results[key] = round(score, 4)
        # 설명: 조건이 맞는지 확인해요.
        if score > best_score:
            # 설명: 값을 저장하거나 바꿔요.
            best_score = score
            # 설명: 값을 저장하거나 바꿔요.
            best_depth = key

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter56",
        # 설명: 다음 코드를 실행해요.
        "topic": "하이퍼파라미터",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "accuracy_by_max_depth": results,
        # 설명: 다음 코드를 실행해요.
        "best_max_depth": best_depth,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
