"""문제 정의서 쓰기 실습 파일"""
from __future__ import annotations


LESSON_10MIN = '무엇을 예측할지(목표), 어떤 입력을 쓸지(특성), 어떤 기준으로 평가할지(지표)를 먼저 정한다.'
PRACTICE_30MIN = "문제 정의서 1장을 딕셔너리로 구성하고 검토 체크를 수행한다."


def run() -> dict:
    problem_card = {
        "project_name": "학생 퀴즈 통과 예측",
        "goal": "다음 퀴즈 통과 여부(0/1) 예측",
        "inputs": ["study_minutes", "attendance_rate", "homework_done"],
        "output": "pass_label",
        "metric": "f1",
        "success_criteria": "f1 >= 0.80",
        "scope": "중학교 1학년 대상 파일럿",
    }

    required_keys = ["goal", "inputs", "output", "metric"]
    missing = [k for k in required_keys if k not in problem_card or not problem_card[k]]

    return {
        "chapter": "chapter89",
        "topic": "문제 정의서 쓰기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "problem_card": problem_card,
        "required_keys": required_keys,
        "missing_keys": missing,
        "is_ready": len(missing) == 0,
    }


if __name__ == "__main__":
    print(run())
