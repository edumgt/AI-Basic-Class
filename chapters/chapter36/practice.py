# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""스케일링 개념 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import StandardScaler


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "서로 단위가 다른 특성은 스케일을 맞추면 학습이 안정적이다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "스케일링 전후 분류 정확도를 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 180

    # 설명: 값을 저장하거나 바꿔요.
    feature_large = rng.normal(loc=5000, scale=900, size=n)
    # 설명: 값을 저장하거나 바꿔요.
    feature_small = rng.normal(loc=0.5, scale=0.12, size=n)
    # 설명: 값을 저장하거나 바꿔요.
    noise = rng.normal(loc=0.0, scale=0.06, size=n)
    # 설명: 값을 저장하거나 바꿔요.
    y = (feature_small + noise > 0.5).astype(int)

    # 설명: 값을 저장하거나 바꿔요.
    X = np.column_stack([feature_large, feature_small])

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    raw_model = LogisticRegression(max_iter=500)
    # 설명: 다음 코드를 실행해요.
    raw_model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    raw_pred = raw_model.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    raw_acc = float(accuracy_score(y_test, raw_pred))

    # 설명: 값을 저장하거나 바꿔요.
    scaler = StandardScaler()
    # 설명: 값을 저장하거나 바꿔요.
    X_train_scaled = scaler.fit_transform(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    X_test_scaled = scaler.transform(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    scaled_model = LogisticRegression(max_iter=500)
    # 설명: 다음 코드를 실행해요.
    scaled_model.fit(X_train_scaled, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    scaled_pred = scaled_model.predict(X_test_scaled)
    # 설명: 값을 저장하거나 바꿔요.
    scaled_acc = float(accuracy_score(y_test, scaled_pred))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter36",
        # 설명: 다음 코드를 실행해요.
        "topic": "스케일링 개념",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 값을 저장하거나 바꿔요.
        "raw_feature_mean": X.mean(axis=0).round(3).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "raw_feature_std": X.std(axis=0).round(3).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "scaled_train_mean": X_train_scaled.mean(axis=0).round(3).tolist(),
        # 설명: 값을 저장하거나 바꿔요.
        "scaled_train_std": X_train_scaled.std(axis=0).round(3).tolist(),
        # 설명: 다음 코드를 실행해요.
        "accuracy_raw": round(raw_acc, 4),
        # 설명: 다음 코드를 실행해요.
        "accuracy_scaled": round(scaled_acc, 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
