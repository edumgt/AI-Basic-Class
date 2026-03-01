"""실험 기록 자동화 실습 파일"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


LESSON_10MIN = "실험 결과를 구조적으로 기록하면 재현성과 협업 품질이 올라간다."
PRACTICE_30MIN = "모델, 파라미터, 지표를 CSV로 저장한다."


def run() -> dict:
    logs = pd.DataFrame(
        [
            {"run_id": "run_001", "model": "logistic", "params": "C=1.0", "accuracy": 0.81, "f1": 0.79},
            {"run_id": "run_002", "model": "logistic", "params": "C=3.0", "accuracy": 0.83, "f1": 0.81},
            {"run_id": "run_003", "model": "random_forest", "params": "n=120", "accuracy": 0.86, "f1": 0.84},
        ]
    )

    out_path = Path(__file__).with_name("experiment_log.csv")
    logs.to_csv(out_path, index=False, encoding="utf-8")

    best_row = logs.sort_values("f1", ascending=False).iloc[0].to_dict()

    return {
        "chapter": "chapter91",
        "topic": "실험 기록 자동화",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "saved_path": str(out_path),
        "rows_saved": int(len(logs)),
        "best_by_f1": best_row,
    }


if __name__ == "__main__":
    print(run())
