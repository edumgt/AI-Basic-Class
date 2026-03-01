# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""간단한 피처 엔지니어링 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LinearRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import r2_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "원본 특성만 쓰는 것보다 새 특성을 만들면 성능이 좋아질 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "파생 특성 추가 전/후 R2를 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 200

    # 설명: 값을 저장하거나 바꿔요.
    study_hours = rng.uniform(1, 6, size=n)
    # 설명: 값을 저장하거나 바꿔요.
    sleep_hours = rng.uniform(5, 9, size=n)

    # 설명: 값을 저장하거나 바꿔요.
    score = 8 * study_hours + 4 * sleep_hours + 2 * study_hours * sleep_hours + rng.normal(0, 2.0, size=n)

    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "study_hours": study_hours,
            # 설명: 다음 코드를 실행해요.
            "sleep_hours": sleep_hours,
            # 설명: 다음 코드를 실행해요.
            "score": score,
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    base_features = ["study_hours", "sleep_hours"]
    # 설명: 값을 저장하거나 바꿔요.
    engineered = df.copy()
    # 설명: 값을 저장하거나 바꿔요.
    engineered["study_x_sleep"] = engineered["study_hours"] * engineered["sleep_hours"]
    # 설명: 값을 저장하거나 바꿔요.
    engineered["study_per_sleep"] = engineered["study_hours"] / engineered["sleep_hours"]

    # 설명: 값을 저장하거나 바꿔요.
    X_base = df[base_features]
    # 설명: 값을 저장하거나 바꿔요.
    X_eng = engineered[["study_hours", "sleep_hours", "study_x_sleep", "study_per_sleep"]]
    # 설명: 값을 저장하거나 바꿔요.
    y = df["score"]

    # 설명: 값을 저장하거나 바꿔요.
    Xb_train, Xb_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X_base, y, test_size=0.25, random_state=42
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    Xe_train, Xe_test, _, _ = train_test_split(X_eng, y, test_size=0.25, random_state=42)

    # 설명: 값을 저장하거나 바꿔요.
    base_model = LinearRegression().fit(Xb_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    eng_model = LinearRegression().fit(Xe_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    base_r2 = float(r2_score(y_test, base_model.predict(Xb_test)))
    # 설명: 값을 저장하거나 바꿔요.
    eng_r2 = float(r2_score(y_test, eng_model.predict(Xe_test)))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter41",
        # 설명: 다음 코드를 실행해요.
        "topic": "간단한 피처 엔지니어링",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "base_features": base_features,
        # 설명: 다음 코드를 실행해요.
        "engineered_features": ["study_x_sleep", "study_per_sleep"],
        # 설명: 다음 코드를 실행해요.
        "r2_base": round(base_r2, 4),
        # 설명: 다음 코드를 실행해요.
        "r2_engineered": round(eng_r2, 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
