# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""데이터 누수 이해 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "미래 정보가 입력에 섞이면 평가 점수가 비정상적으로 높아진다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "누수 특성 포함/제외 정확도를 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    rng = np.random.default_rng(42)
    # 설명: 값을 저장하거나 바꿔요.
    n = 240

    # 설명: 값을 저장하거나 바꿔요.
    hours = rng.normal(5, 1.2, size=n)
    # 설명: 값을 저장하거나 바꿔요.
    attendance = rng.normal(80, 10, size=n)

    # 설명: 값을 저장하거나 바꿔요.
    y = (hours * 0.7 + attendance * 0.03 + rng.normal(0, 0.8, size=n) > 6.0).astype(int)

    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "hours": hours,
            # 설명: 다음 코드를 실행해요.
            "attendance": attendance,
            # 설명: 다음 코드를 실행해요.
            "target": y,
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 의도적인 누수: target을 거의 그대로 보여주는 컬럼
    # 설명: 값을 저장하거나 바꿔요.
    df["future_hint"] = df["target"] + rng.normal(0, 0.02, size=n)

    # 설명: 값을 저장하거나 바꿔요.
    X_safe = df[["hours", "attendance"]]
    # 설명: 값을 저장하거나 바꿔요.
    X_leak = df[["hours", "attendance", "future_hint"]]
    # 설명: 값을 저장하거나 바꿔요.
    y = df["target"]

    # 설명: 값을 저장하거나 바꿔요.
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X_safe, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )
    # 설명: 값을 저장하거나 바꿔요.
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X_leak, y, test_size=0.3, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    safe_model = LogisticRegression(max_iter=500).fit(Xs_train, ys_train)
    # 설명: 값을 저장하거나 바꿔요.
    leak_model = LogisticRegression(max_iter=500).fit(Xl_train, yl_train)

    # 설명: 값을 저장하거나 바꿔요.
    safe_acc = float(accuracy_score(ys_test, safe_model.predict(Xs_test)))
    # 설명: 값을 저장하거나 바꿔요.
    leak_acc = float(accuracy_score(yl_test, leak_model.predict(Xl_test)))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter42",
        # 설명: 다음 코드를 실행해요.
        "topic": "데이터 누수 이해",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "safe_features": ["hours", "attendance"],
        # 설명: 다음 코드를 실행해요.
        "leak_feature": "future_hint",
        # 설명: 다음 코드를 실행해요.
        "accuracy_without_leak": round(safe_acc, 4),
        # 설명: 다음 코드를 실행해요.
        "accuracy_with_leak": round(leak_acc, 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
