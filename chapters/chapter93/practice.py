# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""프론트 연동 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "프론트와 백엔드는 JSON 요청/응답 규칙으로 연결된다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "fetch 예시를 만들고 요청 페이로드를 검증한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    payload = {"study_minutes": 40, "attendance_rate": 0.85, "homework_done": 1}

    # 설명: 값을 저장하거나 바꿔요.
    fetch_example = (
        # 설명: 다음 코드를 실행해요.
        "fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, "
        # 설명: 다음 코드를 실행해요.
        "body: JSON.stringify(payload)})"
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    flow = [
        # 설명: 다음 코드를 실행해요.
        "1) 프론트에서 입력값 수집",
        # 설명: 다음 코드를 실행해요.
        "2) JSON으로 /predict 요청",
        # 설명: 다음 코드를 실행해요.
        "3) 백엔드가 probability/label 반환",
        # 설명: 다음 코드를 실행해요.
        "4) 프론트에서 결과를 화면에 렌더링",
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter93",
        # 설명: 다음 코드를 실행해요.
        "topic": "프론트 연동",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "payload": payload,
        # 설명: 다음 코드를 실행해요.
        "fetch_example": fetch_example,
        # 설명: 다음 코드를 실행해요.
        "integration_flow": flow,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
