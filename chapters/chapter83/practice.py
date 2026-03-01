# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""추천시스템 감각 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "벡터 간 코사인 유사도로 취향이 비슷한 사용자나 아이템을 찾을 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "유사도 점수를 계산해 추천 후보를 만든다."


# 설명: `cosine_similarity` 함수를 만들어요.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 설명: 값을 저장하거나 바꿔요.
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    # 설명: 함수 결과를 돌려줘요.
    return float(np.dot(a, b) / denom)


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 행: 사용자, 열: 아이템
    # 설명: 값을 저장하거나 바꿔요.
    ratings = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [5, 4, 0, 0, 1],
            # 설명: 다음 코드를 실행해요.
            [4, 5, 1, 0, 1],
            # 설명: 다음 코드를 실행해요.
            [0, 1, 5, 4, 0],
            # 설명: 다음 코드를 실행해요.
            [0, 0, 4, 5, 1],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    target_user = 0
    # 설명: 값을 저장하거나 바꿔요.
    sims = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for u in range(ratings.shape[0]):
        # 설명: 조건이 맞는지 확인해요.
        if u == target_user:
            # 설명: 다음 코드를 실행해요.
            continue
        # 설명: 다음 코드를 실행해요.
        sims.append((u, cosine_similarity(ratings[target_user], ratings[u])))

    # 설명: 값을 저장하거나 바꿔요.
    sims.sort(key=lambda x: x[1], reverse=True)
    # 설명: 값을 저장하거나 바꿔요.
    nearest_user = sims[0][0]

    # target 사용자가 아직 안 본 아이템(0)을 nearest user 선호도로 추천
    # 설명: 값을 저장하거나 바꿔요.
    unseen = np.where(ratings[target_user] == 0)[0]
    # 설명: 값을 저장하거나 바꿔요.
    candidate_scores = {int(i): float(ratings[nearest_user, i]) for i in unseen}
    # 설명: 값을 저장하거나 바꿔요.
    recommended_items = [k for k, _ in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter83",
        # 설명: 다음 코드를 실행해요.
        "topic": "추천시스템 감각",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "similarities": [{"user": int(u), "score": round(s, 4)} for u, s in sims],
        # 설명: 다음 코드를 실행해요.
        "nearest_user": int(nearest_user),
        # 설명: 다음 코드를 실행해요.
        "recommended_item_indices": recommended_items,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
