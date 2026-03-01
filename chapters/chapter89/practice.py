# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""문제 정의서 쓰기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = '무엇을 예측할지(목표), 어떤 입력을 쓸지(특성), 어떤 기준으로 평가할지(지표)를 먼저 정한다.'
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "문제 정의서 1장을 딕셔너리로 구성하고 검토 체크를 수행한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    problem_card = {
        # 설명: 다음 코드를 실행해요.
        "project_name": "학생 퀴즈 통과 예측",
        # 설명: 다음 코드를 실행해요.
        "goal": "다음 퀴즈 통과 여부(0/1) 예측",
        # 설명: 다음 코드를 실행해요.
        "inputs": ["study_minutes", "attendance_rate", "homework_done"],
        # 설명: 다음 코드를 실행해요.
        "output": "pass_label",
        # 설명: 다음 코드를 실행해요.
        "metric": "f1",
        # 설명: 값을 저장하거나 바꿔요.
        "success_criteria": "f1 >= 0.80",
        # 설명: 다음 코드를 실행해요.
        "scope": "중학교 1학년 대상 파일럿",
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    required_keys = ["goal", "inputs", "output", "metric"]
    # 설명: 값을 저장하거나 바꿔요.
    missing = [k for k in required_keys if k not in problem_card or not problem_card[k]]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter89",
        # 설명: 다음 코드를 실행해요.
        "topic": "문제 정의서 쓰기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "problem_card": problem_card,
        # 설명: 다음 코드를 실행해요.
        "required_keys": required_keys,
        # 설명: 다음 코드를 실행해요.
        "missing_keys": missing,
        # 설명: 값을 저장하거나 바꿔요.
        "is_ready": len(missing) == 0,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
