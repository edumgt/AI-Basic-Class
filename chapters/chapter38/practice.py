# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""결측치 처리 전략 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.impute import SimpleImputer
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LinearRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import mean_squared_error
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "결측치는 삭제와 대체 중 어떤 전략이 나은지 비교해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "drop, mean, median 전략의 회귀 오차를 비교한다."


# 설명: `_rmse` 함수를 만들어요.
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 220

    # 설명: 값을 저장하거나 바꿔요.
    X = rng.normal(0, 1, size=(n, 3))
    # 설명: 값을 저장하거나 바꿔요.
    y = 12 * X[:, 0] - 4 * X[:, 1] + 2 * X[:, 2] + rng.normal(0, 1.8, size=n)

    # 설명: 값을 저장하거나 바꿔요.
    missing_mask1 = rng.random(n) < 0.18
    # 설명: 값을 저장하거나 바꿔요.
    missing_mask2 = rng.random(n) < 0.12
    # 설명: 값을 저장하거나 바꿔요.
    X[missing_mask1, 0] = np.nan
    # 설명: 값을 저장하거나 바꿔요.
    X[missing_mask2, 1] = np.nan

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.3, random_state=42
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    train_ok = ~np.isnan(X_train).any(axis=1)
    # 설명: 값을 저장하거나 바꿔요.
    test_ok = ~np.isnan(X_test).any(axis=1)

    # 설명: 값을 저장하거나 바꿔요.
    drop_rmse = None
    # 설명: 조건이 맞는지 확인해요.
    if train_ok.sum() > 5 and test_ok.sum() > 1:
        # 설명: 값을 저장하거나 바꿔요.
        drop_model = LinearRegression()
        # 설명: 다음 코드를 실행해요.
        drop_model.fit(X_train[train_ok], y_train[train_ok])
        # 설명: 값을 저장하거나 바꿔요.
        drop_pred = drop_model.predict(X_test[test_ok])
        # 설명: 값을 저장하거나 바꿔요.
        drop_rmse = round(_rmse(y_test[test_ok], drop_pred), 4)

    # 설명: 값을 저장하거나 바꿔요.
    mean_imputer = SimpleImputer(strategy="mean")
    # 설명: 값을 저장하거나 바꿔요.
    median_imputer = SimpleImputer(strategy="median")

    # 설명: 값을 저장하거나 바꿔요.
    X_train_mean = mean_imputer.fit_transform(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    X_test_mean = mean_imputer.transform(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    X_train_median = median_imputer.fit_transform(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    X_test_median = median_imputer.transform(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    mean_model = LinearRegression().fit(X_train_mean, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    median_model = LinearRegression().fit(X_train_median, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred_mean = mean_model.predict(X_test_mean)
    # 설명: 값을 저장하거나 바꿔요.
    pred_median = median_model.predict(X_test_median)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter38",
        # 설명: 다음 코드를 실행해요.
        "topic": "결측치 처리 전략",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "missing_count": int(np.isnan(X).sum()),
        # 설명: 다음 코드를 실행해요.
        "usable_rows_drop": {
            # 설명: 다음 코드를 실행해요.
            "train": int(train_ok.sum()),
            # 설명: 다음 코드를 실행해요.
            "test": int(test_ok.sum()),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "rmse": {
            # 설명: 다음 코드를 실행해요.
            "dropna": drop_rmse,
            # 설명: 다음 코드를 실행해요.
            "mean_impute": round(_rmse(y_test, pred_mean), 4),
            # 설명: 다음 코드를 실행해요.
            "median_impute": round(_rmse(y_test, pred_median), 4),
        # 설명: 다음 코드를 실행해요.
        },
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
