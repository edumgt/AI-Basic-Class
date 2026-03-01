"""모델 배포 맛보기 실습 파일"""
from __future__ import annotations


LESSON_10MIN = "학습 코드와 서비스 코드를 분리하면 운영 중 안정성을 높일 수 있다."
PRACTICE_30MIN = "입력 JSON을 받아 예측값을 반환하는 API 계약을 설계한다."


def predict_pass(study_minutes: float, attendance_rate: float, homework_done: int) -> dict:
    # 간단한 규칙 기반 예측(배포 흐름 데모용)
    score = 0.02 * study_minutes + 2.5 * attendance_rate + 0.8 * homework_done
    prob = min(max(score / 5.0, 0.0), 1.0)
    label = int(prob >= 0.5)
    return {"probability": round(prob, 4), "label": label}


def run() -> dict:
    sample_input = {"study_minutes": 45.0, "attendance_rate": 0.9, "homework_done": 1}
    pred = predict_pass(**sample_input)

    api_contract = {
        "method": "POST",
        "path": "/predict",
        "request_json": {
            "study_minutes": "float",
            "attendance_rate": "float",
            "homework_done": "int(0 or 1)",
        },
        "response_json": {
            "probability": "float(0~1)",
            "label": "int(0 or 1)",
        },
    }

    return {
        "chapter": "chapter92",
        "topic": "모델 배포 맛보기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "api_contract": api_contract,
        "sample_input": sample_input,
        "sample_prediction": pred,
    }


if __name__ == "__main__":
    print(run())
