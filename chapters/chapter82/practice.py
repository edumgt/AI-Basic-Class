"""감성분석 맛보기 실습 파일"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LESSON_10MIN = "감성분석은 문장을 긍정/부정 라벨로 분류하는 문제다."
PRACTICE_30MIN = "작은 문장 데이터로 텍스트 분류기를 학습한다."


def run() -> dict:
    texts = [
        "this movie is great",
        "i love this song",
        "what a wonderful day",
        "this is terrible",
        "i hate this",
        "very bad service",
        "excellent and fun",
        "awful and boring",
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_vec, y_train)

    pred = model.predict(X_test_vec)
    acc = float(accuracy_score(y_test, pred))

    sample_text = "this day is great"
    sample_pred = int(model.predict(vectorizer.transform([sample_text]))[0])

    return {
        "chapter": "chapter82",
        "topic": "감성분석 맛보기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "test_accuracy": round(acc, 4),
        "sample_text": sample_text,
        "sample_prediction": sample_pred,
    }


if __name__ == "__main__":
    print(run())
