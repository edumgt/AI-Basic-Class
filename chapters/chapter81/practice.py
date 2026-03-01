# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""텍스트 데이터 입문 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from collections import Counter


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "문장을 단어 단위로 나누면 숫자 벡터(빈도)로 바꿀 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "단어 사전을 만들고 문장별 카운트 벡터를 생성한다."


# 설명: `_tokenize` 함수를 만들어요.
def _tokenize(text: str) -> list[str]:
    # 설명: 함수 결과를 돌려줘요.
    return text.lower().replace(".", "").replace("!", "").split()


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    sentences = [
        # 설명: 다음 코드를 실행해요.
        "AI class is fun",
        # 설명: 다음 코드를 실행해요.
        "Python class is fun",
        # 설명: 다음 코드를 실행해요.
        "AI and Python are useful",
    # 설명: 다음 코드를 실행해요.
    ]

    # 설명: 값을 저장하거나 바꿔요.
    tokenized = [_tokenize(s) for s in sentences]
    # 설명: 값을 저장하거나 바꿔요.
    vocab = sorted({token for sent in tokenized for token in sent})

    # 설명: 값을 저장하거나 바꿔요.
    vectors = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for sent in tokenized:
        # 설명: 값을 저장하거나 바꿔요.
        cnt = Counter(sent)
        # 설명: 다음 코드를 실행해요.
        vectors.append([cnt[word] for word in vocab])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter81",
        # 설명: 다음 코드를 실행해요.
        "topic": "텍스트 데이터 입문",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "vocab": vocab,
        # 설명: 다음 코드를 실행해요.
        "count_vectors": vectors,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
