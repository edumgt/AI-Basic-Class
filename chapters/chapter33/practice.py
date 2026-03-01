# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_regression
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestRegressor
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LinearRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "같은 데이터로 여러 모델을 비교해 볼 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "선형회귀와 랜덤포레스트의 오차를 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_regression(n_samples=160, n_features=5, noise=14, random_state=42)

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.25, random_state=42
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    linear = LinearRegression()
    # 설명: 값을 저장하거나 바꿔요.
    forest = RandomForestRegressor(n_estimators=120, random_state=42)

    # 설명: 다음 코드를 실행해요.
    linear.fit(X_train, y_train)
    # 설명: 다음 코드를 실행해요.
    forest.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred_linear = linear.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    pred_forest = forest.predict(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    rmse_linear = float(np.sqrt(mean_squared_error(y_test, pred_linear)))
    # 설명: 값을 저장하거나 바꿔요.
    rmse_forest = float(np.sqrt(mean_squared_error(y_test, pred_forest)))

    # 설명: 값을 저장하거나 바꿔요.
    mae_linear = float(mean_absolute_error(y_test, pred_linear))
    # 설명: 값을 저장하거나 바꿔요.
    mae_forest = float(mean_absolute_error(y_test, pred_forest))

    # 설명: 값을 저장하거나 바꿔요.
    better_model = "linear_regression" if rmse_linear <= rmse_forest else "random_forest"

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter33",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "rmse": {
            # 설명: 다음 코드를 실행해요.
            "linear_regression": round(rmse_linear, 4),
            # 설명: 다음 코드를 실행해요.
            "random_forest": round(rmse_forest, 4),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "mae": {
            # 설명: 다음 코드를 실행해요.
            "linear_regression": round(mae_linear, 4),
            # 설명: 다음 코드를 실행해요.
            "random_forest": round(mae_forest, 4),
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "better_model_by_rmse": better_model,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
