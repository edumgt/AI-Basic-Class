"""K-Means 맛보기 실습 파일"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


LESSON_10MIN = "정답 라벨이 없어도 비슷한 데이터끼리 묶을 수 있다."
PRACTICE_30MIN = "2차원 점을 3개 그룹으로 나누고 중심점을 확인한다."


def run() -> dict:
    X = np.array(
        [
            [1.0, 1.1],
            [0.9, 1.0],
            [1.2, 1.3],
            [5.0, 5.1],
            [4.8, 5.0],
            [5.2, 4.9],
            [8.5, 1.2],
            [8.8, 1.0],
            [8.2, 1.3],
        ],
        dtype=float,
    )

    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = model.fit_predict(X)

    cluster_counts = {
        str(cluster_id): int((labels == cluster_id).sum())
        for cluster_id in sorted(np.unique(labels))
    }

    return {
        "chapter": "chapter32",
        "topic": "K-Means 맛보기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "centers": model.cluster_centers_.round(3).tolist(),
        "inertia": round(float(model.inertia_), 4),
    }


if __name__ == "__main__":
    print(run())
