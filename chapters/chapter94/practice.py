# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""오류 처리와 안정성 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "입력 유효성 검사는 서비스 오류와 잘못된 예측을 줄인다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "숫자 범위와 타입 검증 함수를 구현한다."


# 설명: `validate_payload` 함수를 만들어요.
def validate_payload(payload: dict) -> tuple[bool, str]:
    # 설명: 값을 저장하거나 바꿔요.
    required = ["study_minutes", "attendance_rate", "homework_done"]
    # 설명: 같은 동작을 여러 번 반복해요.
    for key in required:
        # 설명: 조건이 맞는지 확인해요.
        if key not in payload:
            # 설명: 함수 결과를 돌려줘요.
            return False, f"missing field: {key}"

    # 설명: 오류가 날 수 있는 동작을 시도해요.
    try:
        # 설명: 값을 저장하거나 바꿔요.
        study = float(payload["study_minutes"])
        # 설명: 값을 저장하거나 바꿔요.
        attend = float(payload["attendance_rate"])
        # 설명: 값을 저장하거나 바꿔요.
        homework = int(payload["homework_done"])
    # 설명: 오류가 나면 안전하게 처리해요.
    except (TypeError, ValueError):
        # 설명: 함수 결과를 돌려줘요.
        return False, "type conversion failed"

    # 설명: 조건이 맞는지 확인해요.
    if not (0 <= study <= 600):
        # 설명: 함수 결과를 돌려줘요.
        return False, "study_minutes out of range"
    # 설명: 조건이 맞는지 확인해요.
    if not (0 <= attend <= 1):
        # 설명: 함수 결과를 돌려줘요.
        return False, "attendance_rate out of range"
    # 설명: 조건이 맞는지 확인해요.
    if homework not in (0, 1):
        # 설명: 함수 결과를 돌려줘요.
        return False, "homework_done must be 0 or 1"

    # 설명: 함수 결과를 돌려줘요.
    return True, "ok"


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    good = {"study_minutes": 45, "attendance_rate": 0.9, "homework_done": 1}
    # 설명: 값을 저장하거나 바꿔요.
    bad = {"study_minutes": -3, "attendance_rate": 1.2, "homework_done": 2}

    # 설명: 값을 저장하거나 바꿔요.
    good_result = validate_payload(good)
    # 설명: 값을 저장하거나 바꿔요.
    bad_result = validate_payload(bad)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter94",
        # 설명: 다음 코드를 실행해요.
        "topic": "오류 처리와 안정성",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "good_payload_check": {"valid": good_result[0], "message": good_result[1]},
        # 설명: 다음 코드를 실행해요.
        "bad_payload_check": {"valid": bad_result[0], "message": bad_result[1]},
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
