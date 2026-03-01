# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""에러 분석 노트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "성능 수치만 보지 말고 어떤 샘플에서 틀렸는지 확인해야 개선 포인트가 보인다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "오분류 샘플을 표로 뽑아 에러 노트를 만든다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=280,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=6,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=4,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=1,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(X, columns=feature_names)
    # 설명: 값을 저장하거나 바꿔요.
    df["target"] = y

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        df[feature_names], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = RandomForestClassifier(n_estimators=160, random_state=42)
    # 설명: 다음 코드를 실행해요.
    model.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    review = X_test.copy().reset_index(drop=True)
    # 설명: 값을 저장하거나 바꿔요.
    review["true"] = y_test.reset_index(drop=True)
    # 설명: 값을 저장하거나 바꿔요.
    review["pred"] = pred
    # 설명: 값을 저장하거나 바꿔요.
    errors = review[review["true"] != review["pred"]].copy()

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter64",
        # 설명: 다음 코드를 실행해요.
        "topic": "에러 분석 노트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "test_rows": int(len(review)),
        # 설명: 다음 코드를 실행해요.
        "error_rows": int(len(errors)),
        # 설명: 다음 코드를 실행해요.
        "error_rate": round(float(len(errors) / len(review)), 4),
        # 설명: 값을 저장하거나 바꿔요.
        "error_samples": errors.head(5).round(4).to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
