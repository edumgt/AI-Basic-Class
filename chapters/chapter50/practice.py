"""회귀 지표 비교 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LESSON_10MIN = "회귀 지표는 같은 예측도 다른 시각으로 평가한다."
PRACTICE_30MIN = "MAE, MSE, RMSE, R2를 두 예측 결과에서 비교한다."


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run() -> dict:
    y_true = np.array([100, 120, 130, 150, 170, 160, 180], dtype=float)
    pred_a = np.array([98, 125, 128, 148, 165, 158, 182], dtype=float)
    pred_b = np.array([90, 135, 120, 155, 178, 150, 190], dtype=float)

    return {
        "chapter": "chapter50",
        "topic": "회귀 지표 비교",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "model_a": _metrics(y_true, pred_a),
        "model_b": _metrics(y_true, pred_b),
    }


if __name__ == "__main__":
    print(run())
