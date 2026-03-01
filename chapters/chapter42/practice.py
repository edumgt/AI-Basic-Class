"""데이터 누수 이해 실습 파일"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "미래 정보가 입력에 섞이면 평가 점수가 비정상적으로 높아진다."
PRACTICE_30MIN = "누수 특성 포함/제외 정확도를 비교한다."


def run() -> dict:
    rng = np.random.default_rng(42)
    n = 240

    hours = rng.normal(5, 1.2, size=n)
    attendance = rng.normal(80, 10, size=n)

    y = (hours * 0.7 + attendance * 0.03 + rng.normal(0, 0.8, size=n) > 6.0).astype(int)

    df = pd.DataFrame(
        {
            "hours": hours,
            "attendance": attendance,
            "target": y,
        }
    )

    # 의도적인 누수: target을 거의 그대로 보여주는 컬럼
    df["future_hint"] = df["target"] + rng.normal(0, 0.02, size=n)

    X_safe = df[["hours", "attendance"]]
    X_leak = df[["hours", "attendance", "future_hint"]]
    y = df["target"]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_safe, y, test_size=0.3, random_state=42, stratify=y
    )
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        X_leak, y, test_size=0.3, random_state=42, stratify=y
    )

    safe_model = LogisticRegression(max_iter=500).fit(Xs_train, ys_train)
    leak_model = LogisticRegression(max_iter=500).fit(Xl_train, yl_train)

    safe_acc = float(accuracy_score(ys_test, safe_model.predict(Xs_test)))
    leak_acc = float(accuracy_score(yl_test, leak_model.predict(Xl_test)))

    return {
        "chapter": "chapter42",
        "topic": "데이터 누수 이해",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "safe_features": ["hours", "attendance"],
        "leak_feature": "future_hint",
        "accuracy_without_leak": round(safe_acc, 4),
        "accuracy_with_leak": round(leak_acc, 4),
    }


if __name__ == "__main__":
    print(run())
