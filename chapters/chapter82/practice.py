# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""감성분석 맛보기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from sklearn.feature_extraction.text import CountVectorizer
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import LogisticRegression
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import train_test_split


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "감성분석은 문장을 긍정/부정 라벨로 분류하는 문제다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "작은 문장 데이터로 텍스트 분류기를 학습한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    texts = [
        # 설명: 다음 코드를 실행해요.
        "this movie is great",
        # 설명: 다음 코드를 실행해요.
        "i love this song",
        # 설명: 다음 코드를 실행해요.
        "what a wonderful day",
        # 설명: 다음 코드를 실행해요.
        "this is terrible",
        # 설명: 다음 코드를 실행해요.
        "i hate this",
        # 설명: 다음 코드를 실행해요.
        "very bad service",
        # 설명: 다음 코드를 실행해요.
        "excellent and fun",
        # 설명: 다음 코드를 실행해요.
        "awful and boring",
    # 설명: 다음 코드를 실행해요.
    ]
    # 설명: 값을 저장하거나 바꿔요.
    labels = [1, 1, 1, 0, 0, 0, 1, 0]

    # 설명: 값을 저장하거나 바꿔요.
    X_train, X_test, y_train, y_test = train_test_split(
        # 설명: 값을 저장하거나 바꿔요.
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    vectorizer = CountVectorizer()
    # 설명: 값을 저장하거나 바꿔요.
    X_train_vec = vectorizer.fit_transform(X_train)
    # 설명: 값을 저장하거나 바꿔요.
    X_test_vec = vectorizer.transform(X_test)

    # 설명: 값을 저장하거나 바꿔요.
    model = LogisticRegression(max_iter=500, random_state=42)
    # 설명: 다음 코드를 실행해요.
    model.fit(X_train_vec, y_train)

    # 설명: 값을 저장하거나 바꿔요.
    pred = model.predict(X_test_vec)
    # 설명: 값을 저장하거나 바꿔요.
    acc = float(accuracy_score(y_test, pred))

    # 설명: 값을 저장하거나 바꿔요.
    sample_text = "this day is great"
    # 설명: 값을 저장하거나 바꿔요.
    sample_pred = int(model.predict(vectorizer.transform([sample_text]))[0])

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter82",
        # 설명: 다음 코드를 실행해요.
        "topic": "감성분석 맛보기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "vocab_size": int(len(vectorizer.vocabulary_)),
        # 설명: 다음 코드를 실행해요.
        "test_accuracy": round(acc, 4),
        # 설명: 다음 코드를 실행해요.
        "sample_text": sample_text,
        # 설명: 다음 코드를 실행해요.
        "sample_prediction": sample_pred,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
