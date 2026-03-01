# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""모델 배포 맛보기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "학습 코드와 서비스 코드를 분리하면 운영 중 안정성을 높일 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "입력 JSON을 받아 예측값을 반환하는 API 계약을 설계한다."


# 설명: `predict_pass` 함수를 만들어요.
def predict_pass(study_minutes: float, attendance_rate: float, homework_done: int) -> dict:
    # 간단한 규칙 기반 예측(배포 흐름 데모용)
    # 설명: 값을 저장하거나 바꿔요.
    score = 0.02 * study_minutes + 2.5 * attendance_rate + 0.8 * homework_done
    # 설명: 값을 저장하거나 바꿔요.
    prob = min(max(score / 5.0, 0.0), 1.0)
    # 설명: 값을 저장하거나 바꿔요.
    label = int(prob >= 0.5)
    # 설명: 함수 결과를 돌려줘요.
    return {"probability": round(prob, 4), "label": label}


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    sample_input = {"study_minutes": 45.0, "attendance_rate": 0.9, "homework_done": 1}
    # 설명: 값을 저장하거나 바꿔요.
    pred = predict_pass(**sample_input)

    # 설명: 값을 저장하거나 바꿔요.
    api_contract = {
        # 설명: 다음 코드를 실행해요.
        "method": "POST",
        # 설명: 다음 코드를 실행해요.
        "path": "/predict",
        # 설명: 다음 코드를 실행해요.
        "request_json": {
            # 설명: 다음 코드를 실행해요.
            "study_minutes": "float",
            # 설명: 다음 코드를 실행해요.
            "attendance_rate": "float",
            # 설명: 다음 코드를 실행해요.
            "homework_done": "int(0 or 1)",
        # 설명: 다음 코드를 실행해요.
        },
        # 설명: 다음 코드를 실행해요.
        "response_json": {
            # 설명: 다음 코드를 실행해요.
            "probability": "float(0~1)",
            # 설명: 다음 코드를 실행해요.
            "label": "int(0 or 1)",
        # 설명: 다음 코드를 실행해요.
        },
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter92",
        # 설명: 다음 코드를 실행해요.
        "topic": "모델 배포 맛보기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "api_contract": api_contract,
        # 설명: 다음 코드를 실행해요.
        "sample_input": sample_input,
        # 설명: 다음 코드를 실행해요.
        "sample_prediction": pred,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
