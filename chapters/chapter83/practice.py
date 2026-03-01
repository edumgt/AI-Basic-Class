"""추천시스템 감각 실습 파일"""
from __future__ import annotations

import numpy as np


LESSON_10MIN = "벡터 간 코사인 유사도로 취향이 비슷한 사용자나 아이템을 찾을 수 있다."
PRACTICE_30MIN = "유사도 점수를 계산해 추천 후보를 만든다."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def run() -> dict:
    # 행: 사용자, 열: 아이템
    ratings = np.array(
        [
            [5, 4, 0, 0, 1],
            [4, 5, 1, 0, 1],
            [0, 1, 5, 4, 0],
            [0, 0, 4, 5, 1],
        ],
        dtype=float,
    )

    target_user = 0
    sims = []
    for u in range(ratings.shape[0]):
        if u == target_user:
            continue
        sims.append((u, cosine_similarity(ratings[target_user], ratings[u])))

    sims.sort(key=lambda x: x[1], reverse=True)
    nearest_user = sims[0][0]

    # target 사용자가 아직 안 본 아이템(0)을 nearest user 선호도로 추천
    unseen = np.where(ratings[target_user] == 0)[0]
    candidate_scores = {int(i): float(ratings[nearest_user, i]) for i in unseen}
    recommended_items = [k for k, _ in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)]

    return {
        "chapter": "chapter83",
        "topic": "추천시스템 감각",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "similarities": [{"user": int(u), "score": round(s, 4)} for u, s in sims],
        "nearest_user": int(nearest_user),
        "recommended_item_indices": recommended_items,
    }


if __name__ == "__main__":
    print(run())
