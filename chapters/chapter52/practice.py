"""과적합과 과소적합 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


LESSON_10MIN = "모델이 너무 단순하면 과소적합, 너무 복잡하면 과적합이 발생할 수 있다."
PRACTICE_30MIN = "다항식 차수를 바꿔 train/test R2를 비교한다."


def _fit_degree(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, degree: int) -> dict[str, float]:
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return {
        "train_r2": round(float(r2_score(y_train, pred_train)), 4),
        "test_r2": round(float(r2_score(y_test, pred_test)), 4),
    }


def run() -> dict:
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 220)
    y = 0.5 * x**3 - 2.0 * x + rng.normal(0, 2.0, size=x.shape[0])

    X = x.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    result = {
        "degree_1": _fit_degree(X_train, X_test, y_train, y_test, degree=1),
        "degree_3": _fit_degree(X_train, X_test, y_train, y_test, degree=3),
        "degree_12": _fit_degree(X_train, X_test, y_train, y_test, degree=12),
    }

    return {
        "chapter": "chapter52",
        "topic": "과적합/과소적합",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "r2_by_degree": result,
    }


if __name__ == "__main__":
    print(run())
