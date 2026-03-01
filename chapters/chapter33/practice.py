"""미니 복습 프로젝트 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


LESSON_10MIN = "같은 데이터로 여러 모델을 비교해 볼 수 있다."
PRACTICE_30MIN = "선형회귀와 랜덤포레스트의 오차를 비교한다."


def run() -> dict:
    X, y = make_regression(n_samples=160, n_features=5, noise=14, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    linear = LinearRegression()
    forest = RandomForestRegressor(n_estimators=120, random_state=42)

    linear.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    pred_linear = linear.predict(X_test)
    pred_forest = forest.predict(X_test)

    rmse_linear = float(np.sqrt(mean_squared_error(y_test, pred_linear)))
    rmse_forest = float(np.sqrt(mean_squared_error(y_test, pred_forest)))

    mae_linear = float(mean_absolute_error(y_test, pred_linear))
    mae_forest = float(mean_absolute_error(y_test, pred_forest))

    better_model = "linear_regression" if rmse_linear <= rmse_forest else "random_forest"

    return {
        "chapter": "chapter33",
        "topic": "미니 복습 프로젝트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "rmse": {
            "linear_regression": round(rmse_linear, 4),
            "random_forest": round(rmse_forest, 4),
        },
        "mae": {
            "linear_regression": round(mae_linear, 4),
            "random_forest": round(mae_forest, 4),
        },
        "better_model_by_rmse": better_model,
    }


if __name__ == "__main__":
    print(run())
