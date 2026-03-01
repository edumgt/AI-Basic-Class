"""생성형 AI 개념 실습 파일"""
from __future__ import annotations

from collections import defaultdict


LESSON_10MIN = "생성형 AI는 정답 분류 대신 새로운 문장이나 이미지를 만들어 낸다."
PRACTICE_30MIN = "아주 단순한 마코프 체인으로 문장 생성을 시도한다."


def _build_bigram_model(corpus: list[str]) -> dict[str, list[str]]:
    model: dict[str, list[str]] = defaultdict(list)
    for sentence in corpus:
        tokens = ["<START>"] + sentence.lower().split() + ["<END>"]
        for a, b in zip(tokens[:-1], tokens[1:]):
            model[a].append(b)
    return model


def _generate(model: dict[str, list[str]], max_len: int = 10) -> str:
    token = "<START>"
    output = []
    for _ in range(max_len):
        next_tokens = sorted(model.get(token, ["<END>"]))
        next_token = next_tokens[0]
        if next_token == "<END>":
            break
        output.append(next_token)
        token = next_token
    return " ".join(output)


def run() -> dict:
    corpus = [
        "ai helps people learn",
        "python helps people build",
        "ai and python build projects",
    ]

    model = _build_bigram_model(corpus)
    generated = _generate(model, max_len=12)

    return {
        "chapter": "chapter86",
        "topic": "생성형 AI 개념",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "model_keys": sorted(model.keys()),
        "generated_sentence": generated,
    }


if __name__ == "__main__":
    print(run())
