# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""과적합과 과소적합 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LinearRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import r2_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import PolynomialFeatures


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "모델이 너무 단순하면 과소적합, 너무 복잡하면 과적합이 발생할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "다항식 차수를 바꿔 train/test R2를 비교한다."


# 설명: `_fit_degree` 함수를 만들어요.
def _fit_degree(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, degree: int) -> dict[str, float]:
    # 설명: 값을 저장하거나 바꿔요.
    model = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 값을 저장하거나 바꿔요.
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            # 설명: 다음 코드를 실행해요.
            ("linear", LinearRegression()),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 다음 코드를 실행해요.
    model.fit(x_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred_train = model.predict(x_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred_test = model.predict(x_test)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "train_r2": round(float(r2_score(y_train, pred_train)), 4),
        # 설명: 다음 코드를 실행해요.
        "test_r2": round(float(r2_score(y_test, pred_test)), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    x = np.linspace(-3, 3, 220)
    # 설명: 값을 저장하거나 바꿔요.
    y = 0.5 * x**3 - 2.0 * x + rng.normal(0, 2.0, size=x.shape[0])

    # 설명: 값을 저장하거나 바꿔요.
    X = x.reshape(-1, 1)
    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 설명: 값을 저장하거나 바꿔요.
    result = {
        # 설명: 값을 저장하거나 바꿔요.
        "degree_1": _fit_degree(X_train, X_test, y_train, y_test, degree=1),
        # 설명: 값을 저장하거나 바꿔요.
        "degree_3": _fit_degree(X_train, X_test, y_train, y_test, degree=3),
        # 설명: 값을 저장하거나 바꿔요.
        "degree_12": _fit_degree(X_train, X_test, y_train, y_test, degree=12),
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter52",
        # 설명: 다음 코드를 실행해요.
        "topic": "과적합/과소적합",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "r2_by_degree": result,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
