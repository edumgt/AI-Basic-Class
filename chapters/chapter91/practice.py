# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""실험 기록 자동화 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from pathlib import Path

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "실험 결과를 구조적으로 기록하면 재현성과 협업 품질이 올라간다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "모델, 파라미터, 지표를 CSV로 저장한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    logs = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 값을 저장하거나 바꿔요.
            {"run_id": "run_001", "model": "logistic", "params": "C=1.0", "accuracy": 0.81, "f1": 0.79},
            # 설명: 값을 저장하거나 바꿔요.
            {"run_id": "run_002", "model": "logistic", "params": "C=3.0", "accuracy": 0.83, "f1": 0.81},
            # 설명: 값을 저장하거나 바꿔요.
            {"run_id": "run_003", "model": "random_forest", "params": "n=120", "accuracy": 0.86, "f1": 0.84},
        # 설명: 다음 코드를 실행해요.
        ]
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    out_path = Path(__file__).with_name("experiment_log.csv")
    # 설명: 값을 저장하거나 바꿔요.
    logs.to_csv(out_path, index=False, encoding="utf-8")

    # 설명: 값을 저장하거나 바꿔요.
    best_row = logs.sort_values("f1", ascending=False).iloc[0].to_dict()

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter91",
        # 설명: 다음 코드를 실행해요.
        "topic": "실험 기록 자동화",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "saved_path": str(out_path),
        # 설명: 다음 코드를 실행해요.
        "rows_saved": int(len(logs)),
        # 설명: 다음 코드를 실행해요.
        "best_by_f1": best_row,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
