"""과적합 방지(정규화/드롭아웃 개념) 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


LESSON_10MIN = "복잡한 모델은 과적합되기 쉬우며 정규화가 일반화 성능을 도울 수 있다."
PRACTICE_30MIN = "고차 다항식 회귀에서 정규화 유무에 따른 train/test 오차를 비교한다."


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def run() -> dict:
    rng = np.random.default_rng(42)

    x = np.linspace(-3, 3, 180)
    y = np.sin(1.3 * x) + rng.normal(0, 0.12, size=x.shape[0])
    X = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    no_reg = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )

    ridge_reg = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    no_reg.fit(X_train, y_train)
    ridge_reg.fit(X_train, y_train)

    pred_train_no = no_reg.predict(X_train)
    pred_test_no = no_reg.predict(X_test)
    pred_train_ridge = ridge_reg.predict(X_train)
    pred_test_ridge = ridge_reg.predict(X_test)

    train_no = _mse(y_train, pred_train_no)
    test_no = _mse(y_test, pred_test_no)
    train_ridge = _mse(y_train, pred_train_ridge)
    test_ridge = _mse(y_test, pred_test_ridge)

    return {
        "chapter": "chapter76",
        "topic": "과적합 방지(정규화/드롭아웃 개념)",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "no_regularization": {
            "train_mse": round(train_no, 6),
            "test_mse": round(test_no, 6),
            "generalization_gap": round(test_no - train_no, 6),
        },
        "ridge_regularization": {
            "train_mse": round(train_ridge, 6),
            "test_mse": round(test_ridge, 6),
            "generalization_gap": round(test_ridge - train_ridge, 6),
        },
    }


if __name__ == "__main__":
    print(run())
