"""결측치 처리 전략 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


LESSON_10MIN = "결측치는 삭제와 대체 중 어떤 전략이 나은지 비교해야 한다."
PRACTICE_30MIN = "drop, mean, median 전략의 회귀 오차를 비교한다."


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 220

    X = rng.normal(0, 1, size=(n, 3))
    y = 12 * X[:, 0] - 4 * X[:, 1] + 2 * X[:, 2] + rng.normal(0, 1.8, size=n)

    missing_mask1 = rng.random(n) < 0.18
    missing_mask2 = rng.random(n) < 0.12
    X[missing_mask1, 0] = np.nan
    X[missing_mask2, 1] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_ok = ~np.isnan(X_train).any(axis=1)
    test_ok = ~np.isnan(X_test).any(axis=1)

    drop_rmse = None
    if train_ok.sum() > 5 and test_ok.sum() > 1:
        drop_model = LinearRegression()
        drop_model.fit(X_train[train_ok], y_train[train_ok])
        drop_pred = drop_model.predict(X_test[test_ok])
        drop_rmse = round(_rmse(y_test[test_ok], drop_pred), 4)

    mean_imputer = SimpleImputer(strategy="mean")
    median_imputer = SimpleImputer(strategy="median")

    X_train_mean = mean_imputer.fit_transform(X_train)
    X_test_mean = mean_imputer.transform(X_test)
    X_train_median = median_imputer.fit_transform(X_train)
    X_test_median = median_imputer.transform(X_test)

    mean_model = LinearRegression().fit(X_train_mean, y_train)
    median_model = LinearRegression().fit(X_train_median, y_train)

    pred_mean = mean_model.predict(X_test_mean)
    pred_median = median_model.predict(X_test_median)

    return {
        "chapter": "chapter38",
        "topic": "결측치 처리 전략",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "missing_count": int(np.isnan(X).sum()),
        "usable_rows_drop": {
            "train": int(train_ok.sum()),
            "test": int(test_ok.sum()),
        },
        "rmse": {
            "dropna": drop_rmse,
            "mean_impute": round(_rmse(y_test, pred_mean), 4),
            "median_impute": round(_rmse(y_test, pred_median), 4),
        },
    }


if __name__ == "__main__":
    print(run())
