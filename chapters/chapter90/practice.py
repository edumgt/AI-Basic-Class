"""데이터 수집 체크리스트 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "좋은 데이터는 양뿐 아니라 대표성, 품질, 누락 여부를 함께 점검해야 한다."
PRACTICE_30MIN = "샘플 데이터로 결측치, 중복, 클래스 비율 진단표를 만든다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4, 4, 5, 6, 7],
            "study_minutes": [30, 45, None, 20, 20, 55, 40, None],
            "attendance_rate": [0.9, 0.8, 0.95, 0.7, 0.7, 0.98, 0.85, 0.6],
            "pass_label": [1, 1, 1, 0, 0, 1, 1, 0],
        }
    )

    diagnostics = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_ratio": {str(k): round(float(v), 4) for k, v in df["pass_label"].value_counts(normalize=True).to_dict().items()},
    }

    checklist = {
        "enough_rows": diagnostics["rows"] >= 100,
        "low_missing": diagnostics["missing_total"] <= 2,
        "low_duplicates": diagnostics["duplicate_rows"] == 0,
        "balanced_target": 0.2 <= diagnostics["target_ratio"].get("1", 0.0) <= 0.8,
    }

    return {
        "chapter": "chapter90",
        "topic": "데이터 수집 체크리스트",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "diagnostics": diagnostics,
        "checklist_pass": checklist,
        "ready_for_training": all(checklist.values()),
    }


if __name__ == "__main__":
    print(run())
