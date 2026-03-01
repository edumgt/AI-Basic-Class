# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""미니 복습 프로젝트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import RandomForestClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import f1_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.tree import DecisionTreeClassifier


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "기준 모델과 개선 모델을 같은 테스트셋에서 비교해야 개선 폭을 신뢰할 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "baseline 대비 개선률을 계산해 보고서 형태로 출력한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X, y = make_classification(
        # 설명: 값을 저장하거나 바꿔요.
        n_samples=420,
        # 설명: 값을 저장하거나 바꿔요.
        n_features=12,
        # 설명: 값을 저장하거나 바꿔요.
        n_informative=7,
        # 설명: 값을 저장하거나 바꿔요.
        n_redundant=2,
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
    baseline = DecisionTreeClassifier(max_depth=3, random_state=42)
    # 설명: 값을 저장하거나 바꿔요.
    improved = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

    # 설명: 다음 코드를 실행해요.
    baseline.fit(X_train, y_train)
    # 설명: 다음 코드를 실행해요.
    improved.fit(X_train, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    f1_base = float(f1_score(y_test, baseline.predict(X_test)))
    # 설명: 값을 저장하거나 바꿔요.
    f1_improved = float(f1_score(y_test, improved.predict(X_test)))

    # 설명: 값을 저장하거나 바꿔요.
    improvement_pct = float((f1_improved - f1_base) / max(abs(f1_base), 1e-9) * 100)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter66",
        # 설명: 다음 코드를 실행해요.
        "topic": "미니 복습 프로젝트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "baseline_f1": round(f1_base, 4),
        # 설명: 다음 코드를 실행해요.
        "improved_f1": round(f1_improved, 4),
        # 설명: 다음 코드를 실행해요.
        "improvement_percent": round(improvement_pct, 2),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
