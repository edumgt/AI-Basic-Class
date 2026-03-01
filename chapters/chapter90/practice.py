# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""데이터 수집 체크리스트 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "좋은 데이터는 양뿐 아니라 대표성, 품질, 누락 여부를 함께 점검해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "샘플 데이터로 결측치, 중복, 클래스 비율 진단표를 만든다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "student_id": [1, 2, 3, 4, 4, 5, 6, 7],
            # 설명: 다음 코드를 실행해요.
            "study_minutes": [30, 45, None, 20, 20, 55, 40, None],
            # 설명: 다음 코드를 실행해요.
            "attendance_rate": [0.9, 0.8, 0.95, 0.7, 0.7, 0.98, 0.85, 0.6],
            # 설명: 다음 코드를 실행해요.
            "pass_label": [1, 1, 1, 0, 0, 1, 1, 0],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    diagnostics = {
        # 설명: 다음 코드를 실행해요.
        "rows": int(len(df)),
        # 설명: 다음 코드를 실행해요.
        "columns": int(df.shape[1]),
        # 설명: 다음 코드를 실행해요.
        "missing_total": int(df.isna().sum().sum()),
        # 설명: 다음 코드를 실행해요.
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        # 설명: 다음 코드를 실행해요.
        "duplicate_rows": int(df.duplicated().sum()),
        # 설명: 값을 저장하거나 바꿔요.
        "target_ratio": {str(k): round(float(v), 4) for k, v in df["pass_label"].value_counts(normalize=True).to_dict().items()},
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    checklist = {
        # 설명: 값을 저장하거나 바꿔요.
        "enough_rows": diagnostics["rows"] >= 100,
        # 설명: 값을 저장하거나 바꿔요.
        "low_missing": diagnostics["missing_total"] <= 2,
        # 설명: 값을 저장하거나 바꿔요.
        "low_duplicates": diagnostics["duplicate_rows"] == 0,
        # 설명: 값을 저장하거나 바꿔요.
        "balanced_target": 0.2 <= diagnostics["target_ratio"].get("1", 0.0) <= 0.8,
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter90",
        # 설명: 다음 코드를 실행해요.
        "topic": "데이터 수집 체크리스트",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "diagnostics": diagnostics,
        # 설명: 다음 코드를 실행해요.
        "checklist_pass": checklist,
        # 설명: 다음 코드를 실행해요.
        "ready_for_training": all(checklist.values()),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
