"""프롬프트 기초 실습 파일"""
from __future__ import annotations


LESSON_10MIN = "좋은 프롬프트는 역할, 목표, 형식, 제약을 명확히 담는다."
PRACTICE_30MIN = "같은 작업을 3가지 프롬프트로 비교하고 구조 점수를 확인한다."


def _prompt_score(prompt: str) -> int:
    p = prompt.lower()
    score = 0
    if "role" in p or "역할" in p:
        score += 1
    if "goal" in p or "목표" in p:
        score += 1
    if "format" in p or "형식" in p:
        score += 1
    if "limit" in p or "제약" in p:
        score += 1
    return score


def run() -> dict:
    prompts = {
        "basic": "서울 맛집 알려줘",
        "better": "역할: 여행 가이드. 목표: 서울 점심 맛집 3개 추천. 형식: 표.",
        "best": "역할: 여행 가이드. 목표: 서울 점심 맛집 3개 추천. 형식: 표(이름/메뉴/가격). 제약: 1인 2만원 이하, 2026년 기준 인기 지역 중심.",
    }

    scored = [
        {
            "label": name,
            "score": _prompt_score(text),
            "prompt": text,
        }
        for name, text in prompts.items()
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "chapter": "chapter87",
        "topic": "프롬프트 기초",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "ranked_prompts": scored,
    }


if __name__ == "__main__":
    print(run())
