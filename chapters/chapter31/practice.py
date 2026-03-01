"""KNN 맛보기 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


LESSON_10MIN = "가까운 점(이웃)을 기준으로 분류할 수 있다."
PRACTICE_30MIN = "K 값을 바꿔 정확도가 어떻게 변하는지 비교한다."


def run() -> dict:
    X = np.array(
        [
            [1.0, 1.1],
            [1.2, 0.9],
            [0.8, 1.0],
            [3.8, 4.2],
            [4.1, 3.9],
            [3.9, 4.0],
            [1.1, 1.3],
            [4.0, 3.7],
            [0.9, 1.2],
            [4.2, 4.1],
            [1.3, 1.0],
            [3.7, 4.3],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    scores: dict[str, float] = {}
    best_k = 1
    best_score = -1.0

    for k in range(1, 6):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = float(accuracy_score(y_test, pred))
        scores[f"k={k}"] = round(score, 4)
        if score > best_score:
            best_score = score
            best_k = k

    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    new_point = np.array([[1.05, 1.15]], dtype=float)
    new_pred = int(best_model.predict(new_point)[0])

    return {
        "chapter": "chapter31",
        "topic": "KNN 맛보기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "k_scores": scores,
        "best_k": best_k,
        "test_size": int(len(X_test)),
        "new_point": new_point.flatten().round(2).tolist(),
        "new_point_prediction": new_pred,
    }


if __name__ == "__main__":
    print(run())
