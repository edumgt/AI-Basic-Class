# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""최종 미니 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score, f1_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import StandardScaler


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "최종 프로젝트는 문제정의-전처리-학습-평가-보고까지 한 흐름으로 완성한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "합성 데이터 하나로 end-to-end 분류 파이프라인을 실행한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 260

    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 값을 저장하거나 바꿔요.
            "study_minutes": rng.normal(45, 15, size=n),
            # 설명: 값을 저장하거나 바꿔요.
            "attendance_rate": rng.uniform(0.6, 1.0, size=n),
            # 설명: 값을 저장하거나 바꿔요.
            "quiz_score": rng.normal(70, 12, size=n),
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    y = (
        # 설명: 다음 코드를 실행해요.
        0.02 * df["study_minutes"]
        # 설명: 다음 코드를 실행해요.
        + 1.8 * df["attendance_rate"]
        # 설명: 다음 코드를 실행해요.
        + 0.03 * df["quiz_score"]
        # 설명: 값을 저장하거나 바꿔요.
        + rng.normal(0, 0.35, size=n)
        # 설명: 다음 코드를 실행해요.
        > 4.2
    # 설명: 다음 코드를 실행해요.
    ).astype(int)

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        df, y, test_size=0.25, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            ("scaler", StandardScaler()),
            # 설명: 값을 저장하거나 바꿔요.
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter98",
        # 설명: 다음 코드를 실행해요.
        "topic": "최종 미니 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "train_rows": int(len(X_train)),
        # 설명: 다음 코드를 실행해요.
        "test_rows": int(len(X_test)),
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        # 설명: 다음 코드를 실행해요.
        "f1": round(float(f1_score(y_test, pred)), 4),
        # 설명: 다음 코드를 실행해요.
        "pipeline_steps": ["scaler", "classifier"],
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
