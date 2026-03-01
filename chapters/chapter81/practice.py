"""텍스트 데이터 입문 실습 파일"""
from __future__ import annotations

from collections import Counter


LESSON_10MIN = "문장을 단어 단위로 나누면 숫자 벡터(빈도)로 바꿀 수 있다."
PRACTICE_30MIN = "단어 사전을 만들고 문장별 카운트 벡터를 생성한다."


def _tokenize(text: str) -> list[str]:
    return text.lower().replace(".", "").replace("!", "").split()


def run() -> dict:
    sentences = [
        "AI class is fun",
        "Python class is fun",
        "AI and Python are useful",
    ]

    tokenized = [_tokenize(s) for s in sentences]
    vocab = sorted({token for sent in tokenized for token in sent})

    vectors = []
    for sent in tokenized:
        cnt = Counter(sent)
        vectors.append([cnt[word] for word in vocab])

    return {
        "chapter": "chapter81",
        "topic": "텍스트 데이터 입문",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "vocab": vocab,
        "count_vectors": vectors,
    }


if __name__ == "__main__":
    print(run())
