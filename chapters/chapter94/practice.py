"""오류 처리와 안정성 실습 파일"""
from __future__ import annotations


LESSON_10MIN = "입력 유효성 검사는 서비스 오류와 잘못된 예측을 줄인다."
PRACTICE_30MIN = "숫자 범위와 타입 검증 함수를 구현한다."


def validate_payload(payload: dict) -> tuple[bool, str]:
    required = ["study_minutes", "attendance_rate", "homework_done"]
    for key in required:
        if key not in payload:
            return False, f"missing field: {key}"

    try:
        study = float(payload["study_minutes"])
        attend = float(payload["attendance_rate"])
        homework = int(payload["homework_done"])
    except (TypeError, ValueError):
        return False, "type conversion failed"

    if not (0 <= study <= 600):
        return False, "study_minutes out of range"
    if not (0 <= attend <= 1):
        return False, "attendance_rate out of range"
    if homework not in (0, 1):
        return False, "homework_done must be 0 or 1"

    return True, "ok"


def run() -> dict:
    good = {"study_minutes": 45, "attendance_rate": 0.9, "homework_done": 1}
    bad = {"study_minutes": -3, "attendance_rate": 1.2, "homework_done": 2}

    good_result = validate_payload(good)
    bad_result = validate_payload(bad)

    return {
        "chapter": "chapter94",
        "topic": "오류 처리와 안정성",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "good_payload_check": {"valid": good_result[0], "message": good_result[1]},
        "bad_payload_check": {"valid": bad_result[0], "message": bad_result[1]},
    }


if __name__ == "__main__":
    print(run())
