# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""과적합 방지(정규화/드롭아웃 개념) 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LinearRegression, Ridge
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import mean_squared_error
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import PolynomialFeatures


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "복잡한 모델은 과적합되기 쉬우며 정규화가 일반화 성능을 도울 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "고차 다항식 회귀에서 정규화 유무에 따른 train/test 오차를 비교한다."


# 설명: `_mse` 함수를 만들어요.
def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return float(mean_squared_error(y_true, y_pred))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)

    # 설명: 값을 저장하거나 바꿔요.
    x = np.linspace(-3, 3, 180)
    # 설명: 값을 저장하거나 바꿔요.
    y = np.sin(1.3 * x) + rng.normal(0, 0.12, size=x.shape[0])
    # 설명: 값을 저장하거나 바꿔요.
    X = x.reshape(-1, 1)

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    no_reg = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 값을 저장하거나 바꿔요.
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            # 설명: 다음 코드를 실행해요.
            ("model", LinearRegression()),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    ridge_reg = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 값을 저장하거나 바꿔요.
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            # 설명: 값을 저장하거나 바꿔요.
            ("model", Ridge(alpha=1.0)),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 다음 코드를 실행해요.
    no_reg.fit(X_train, y_train)
    # 설명: 다음 코드를 실행해요.
    ridge_reg.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred_train_no = no_reg.predict(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred_test_no = no_reg.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    pred_train_ridge = ridge_reg.predict(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred_test_ridge = ridge_reg.predict(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    train_no = _mse(y_train, pred_train_no)
    # 설명: 값을 저장하거나 바꿔요.
    test_no = _mse(y_test, pred_test_no)
    # 설명: 값을 저장하거나 바꿔요.
    train_ridge = _mse(y_train, pred_train_ridge)
    # 설명: 값을 저장하거나 바꿔요.
    test_ridge = _mse(y_test, pred_test_ridge)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter76",
        # 설명: 다음 코드를 실행해요.
        "topic": "과적합 방지(정규화/드롭아웃 개념)",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "no_regularization": {
            # 설명: 다음 코드를 실행해요.
            "train_mse": round(train_no, 6),
            # 설명: 다음 코드를 실행해요.
            "test_mse": round(test_no, 6),
            # 설명: 다음 코드를 실행해요.
            "generalization_gap": round(test_no - train_no, 6),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "ridge_regularization": {
            # 설명: 다음 코드를 실행해요.
            "train_mse": round(train_ridge, 6),
            # 설명: 다음 코드를 실행해요.
            "test_mse": round(test_ridge, 6),
            # 설명: 다음 코드를 실행해요.
            "generalization_gap": round(test_ridge - train_ridge, 6),
        # 설명: 다음 코드를 실행해요.
        },
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
