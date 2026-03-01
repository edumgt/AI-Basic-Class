# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""파이프라인 기초 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import StandardScaler


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "전처리와 모델을 파이프라인으로 묶으면 실수를 줄일 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "StandardScaler와 LogisticRegression을 한 줄 흐름으로 실행한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=220,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=8,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=5,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=1,
        # 설명: 값을 저장하거나 바꿔요.
        random_state=42,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        X, y, test_size=0.25, random_state=42, stratify=y
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    pipe = Pipeline(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            ("scaler", StandardScaler()),
            # 설명: 값을 저장하거나 바꿔요.
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 다음 코드를 실행해요.
    pipe.fit(X_train, y_train)
    # 설명: 값을 저장하거나 바꿔요.
    pred = pipe.predict(X_test)
    # 설명: 값을 저장하거나 바꿔요.
    acc = float(accuracy_score(y_test, pred))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter37",
        # 설명: 다음 코드를 실행해요.
        "topic": "파이프라인 기초",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "pipeline_steps": list(pipe.named_steps.keys()),
        # 설명: 다음 코드를 실행해요.
        "accuracy": round(acc, 4),
        # 설명: 다음 코드를 실행해요.
        "sample_predictions": pred[:10].tolist(),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
