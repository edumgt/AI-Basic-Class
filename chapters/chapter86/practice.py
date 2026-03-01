# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""생성형 AI 개념 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from collections import defaultdict


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "생성형 AI는 정답 분류 대신 새로운 문장이나 이미지를 만들어 낸다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "아주 단순한 마코프 체인으로 문장 생성을 시도한다."


# 설명: `_build_bigram_model` 함수를 만들어요.
def _build_bigram_model(corpus: list[str]) -> dict[str, list[str]]:
    # 설명: 값을 저장하거나 바꿔요.
    model: dict[str, list[str]] = defaultdict(list)
    # 설명: 같은 동작을 여러 번 반복해요.
    for sentence in corpus:
        # 설명: 값을 저장하거나 바꿔요.
        tokens = ["<START>"] + sentence.lower().split() + ["<END>"]
        # 설명: 같은 동작을 여러 번 반복해요.
        for a, b in zip(tokens[:-1], tokens[1:]):
            # 설명: 다음 코드를 실행해요.
            model[a].append(b)
    # 설명: 함수 결과를 돌려줘요.
    return model


# 설명: `_generate` 함수를 만들어요.
def _generate(model: dict[str, list[str]], max_len: int = 10) -> str:
    # 설명: 값을 저장하거나 바꿔요.
    token = "<START>"
    # 설명: 값을 저장하거나 바꿔요.
    output = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(max_len):
        # 설명: 값을 저장하거나 바꿔요.
        next_tokens = sorted(model.get(token, ["<END>"]))
        # 설명: 값을 저장하거나 바꿔요.
        next_token = next_tokens[0]
        # 설명: 조건이 맞는지 확인해요.
        if next_token == "<END>":
            # 설명: 다음 코드를 실행해요.
            break
        # 설명: 다음 코드를 실행해요.
        output.append(next_token)
        # 설명: 값을 저장하거나 바꿔요.
        token = next_token
    # 설명: 함수 결과를 돌려줘요.
    return " ".join(output)


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    corpus = [
        # 설명: 다음 코드를 실행해요.
        "ai helps people learn",
        # 설명: 다음 코드를 실행해요.
        "python helps people build",
        # 설명: 다음 코드를 실행해요.
        "ai and python build projects",
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 값을 저장하거나 바꿔요.
    model = _build_bigram_model(corpus)
    # 설명: 값을 저장하거나 바꿔요.
    generated = _generate(model, max_len=12)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter86",
        # 설명: 다음 코드를 실행해요.
        "topic": "생성형 AI 개념",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "model_keys": sorted(model.keys()),
        # 설명: 다음 코드를 실행해요.
        "generated_sentence": generated,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
