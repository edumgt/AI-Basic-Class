"""프론트 연동 실습 파일"""
from __future__ import annotations


LESSON_10MIN = "프론트와 백엔드는 JSON 요청/응답 규칙으로 연결된다."
PRACTICE_30MIN = "fetch 예시를 만들고 요청 페이로드를 검증한다."


def run() -> dict:
    payload = {"study_minutes": 40, "attendance_rate": 0.85, "homework_done": 1}

    fetch_example = (
        "fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, "
        "body: JSON.stringify(payload)})"
    )

    flow = [
        "1) 프론트에서 입력값 수집",
        "2) JSON으로 /predict 요청",
        "3) 백엔드가 probability/label 반환",
        "4) 프론트에서 결과를 화면에 렌더링",
    ]

    return {
        "chapter": "chapter93",
        "topic": "프론트 연동",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "payload": payload,
        "fetch_example": fetch_example,
        "integration_flow": flow,
    }


if __name__ == "__main__":
    print(run())
