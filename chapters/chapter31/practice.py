# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""KNN 맛보기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.neighbors import KNeighborsClassifier


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "가까운 점(이웃)을 기준으로 분류할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "K 값을 바꿔 정확도가 어떻게 변하는지 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [1.0, 1.1],
            # 설명: 다음 코드를 실행해요.
            [1.2, 0.9],
            # 설명: 다음 코드를 실행해요.
            [0.8, 1.0],
            # 설명: 다음 코드를 실행해요.
            [3.8, 4.2],
            # 설명: 다음 코드를 실행해요.
            [4.1, 3.9],
            # 설명: 다음 코드를 실행해요.
            [3.9, 4.0],
            # 설명: 다음 코드를 실행해요.
            [1.1, 1.3],
            # 설명: 다음 코드를 실행해요.
            [4.0, 3.7],
            # 설명: 다음 코드를 실행해요.
            [0.9, 1.2],
            # 설명: 다음 코드를 실행해요.
            [4.2, 4.1],
            # 설명: 다음 코드를 실행해요.
            [1.3, 1.0],
            # 설명: 다음 코드를 실행해요.
            [3.7, 4.3],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.33, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    scores: dict[str, float] = {}
    # 설명: 값을 저장하거나 바꿔요.
    best_k = 1
    # 설명: 값을 저장하거나 바꿔요.
    best_score = -1.0

    # 설명: 같은 동작을 여러 번 반복해요.
    for k in range(1, 6):
        # 설명: 값을 저장하거나 바꿔요.
        model = KNeighborsClassifier(n_neighbors=k)
        # 설명: 다음 코드를 실행해요.
        model.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        score = float(accuracy_score(y_test, pred))
        # 설명: 값을 저장하거나 바꿔요.
        scores[f"k={k}"] = round(score, 4)
        # 설명: 조건이 맞는지 확인해요.
        if score > best_score:
            # 설명: 값을 저장하거나 바꿔요.
            best_score = score
            # 설명: 값을 저장하거나 바꿔요.
            best_k = k

    # 설명: 값을 저장하거나 바꿔요.
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    # 설명: 다음 코드를 실행해요.
    best_model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    new_point = np.array([[1.05, 1.15]], dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    new_pred = int(best_model.predict(new_point)[0])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter31",
        # 설명: 다음 코드를 실행해요.
        "topic": "KNN 맛보기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "k_scores": scores,
        # 설명: 다음 코드를 실행해요.
        "best_k": best_k,
        # 설명: 다음 코드를 실행해요.
        "test_size": int(len(X_test)),
        # 설명: 다음 코드를 실행해요.
        "new_point": new_point.flatten().round(2).tolist(),
        # 설명: 다음 코드를 실행해요.
        "new_point_prediction": new_pred,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
