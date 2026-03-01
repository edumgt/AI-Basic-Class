# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""회귀 지표 비교 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "회귀 지표는 같은 예측도 다른 시각으로 평가한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "MAE, MSE, RMSE, R2를 두 예측 결과에서 비교한다."


# 설명: `_metrics` 함수를 만들어요.
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # 설명: 값을 저장하거나 바꿔요.
    mse = float(mean_squared_error(y_true, y_pred))
    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        # 설명: 다음 코드를 실행해요.
        "mse": round(mse, 4),
        # 설명: 다음 코드를 실행해요.
        "rmse": round(float(np.sqrt(mse)), 4),
        # 설명: 다음 코드를 실행해요.
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    y_true = np.array([100, 120, 130, 150, 170, 160, 180], dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    pred_a = np.array([98, 125, 128, 148, 165, 158, 182], dtype=float)
    # 설명: 값을 저장하거나 바꿔요.
    pred_b = np.array([90, 135, 120, 155, 178, 150, 190], dtype=float)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter50",
        # 설명: 다음 코드를 실행해요.
        "topic": "회귀 지표 비교",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "model_a": _metrics(y_true, pred_a),
        # 설명: 다음 코드를 실행해요.
        "model_b": _metrics(y_true, pred_b),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
