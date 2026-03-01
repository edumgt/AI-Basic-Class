# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""프롬프트 기초 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "좋은 프롬프트는 역할, 목표, 형식, 제약을 명확히 담는다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "같은 작업을 3가지 프롬프트로 비교하고 구조 점수를 확인한다."


# 설명: `_prompt_score` 함수를 만들어요.
def _prompt_score(prompt: str) -> int:
    # 설명: 값을 저장하거나 바꿔요.
    p = prompt.lower()
    # 설명: 값을 저장하거나 바꿔요.
    score = 0
    # 설명: 조건이 맞는지 확인해요.
    if "role" in p or "역할" in p:
        # 설명: 값을 저장하거나 바꿔요.
        score += 1
    # 설명: 조건이 맞는지 확인해요.
    if "goal" in p or "목표" in p:
        # 설명: 값을 저장하거나 바꿔요.
        score += 1
    # 설명: 조건이 맞는지 확인해요.
    if "format" in p or "형식" in p:
        # 설명: 값을 저장하거나 바꿔요.
        score += 1
    # 설명: 조건이 맞는지 확인해요.
    if "limit" in p or "제약" in p:
        # 설명: 값을 저장하거나 바꿔요.
        score += 1
    # 설명: 함수 결과를 돌려줘요.
    return score


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    prompts = {
        # 설명: 다음 코드를 실행해요.
        "basic": "서울 맛집 알려줘",
        # 설명: 다음 코드를 실행해요.
        "better": "역할: 여행 가이드. 목표: 서울 점심 맛집 3개 추천. 형식: 표.",
        # 설명: 다음 코드를 실행해요.
        "best": "역할: 여행 가이드. 목표: 서울 점심 맛집 3개 추천. 형식: 표(이름/메뉴/가격). 제약: 1인 2만원 이하, 2026년 기준 인기 지역 중심.",
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 값을 저장하거나 바꿔요.
    scored = [
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "label": name,
            # 설명: 다음 코드를 실행해요.
            "score": _prompt_score(text),
            # 설명: 다음 코드를 실행해요.
            "prompt": text,
        # 설명: 다음 코드를 실행해요.
        }
        # 설명: 같은 동작을 여러 번 반복해요.
        for name, text in prompts.items()
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 값을 저장하거나 바꿔요.
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter87",
        # 설명: 다음 코드를 실행해요.
        "topic": "프롬프트 기초",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "ranked_prompts": scored,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
