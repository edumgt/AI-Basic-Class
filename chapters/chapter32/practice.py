# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""K-Means 맛보기 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
from sklearn.cluster import KMeans


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "정답 라벨이 없어도 비슷한 데이터끼리 묶을 수 있다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "2차원 점을 3개 그룹으로 나누고 중심점을 확인한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    X = np.array(
        # 설명: 다음 코드를 실행해요.
        [
            # 설명: 다음 코드를 실행해요.
            [1.0, 1.1],
            # 설명: 다음 코드를 실행해요.
            [0.9, 1.0],
            # 설명: 다음 코드를 실행해요.
            [1.2, 1.3],
            # 설명: 다음 코드를 실행해요.
            [5.0, 5.1],
            # 설명: 다음 코드를 실행해요.
            [4.8, 5.0],
            # 설명: 다음 코드를 실행해요.
            [5.2, 4.9],
            # 설명: 다음 코드를 실행해요.
            [8.5, 1.2],
            # 설명: 다음 코드를 실행해요.
            [8.8, 1.0],
            # 설명: 다음 코드를 실행해요.
            [8.2, 1.3],
        # 설명: 다음 코드를 실행해요.
        ],
        # 설명: 값을 저장하거나 바꿔요.
        dtype=float,
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    # 설명: 값을 저장하거나 바꿔요.
    labels = model.fit_predict(X)

    # 설명: 값을 저장하거나 바꿔요.
    cluster_counts = {
        # 설명: 값을 저장하거나 바꿔요.
        str(cluster_id): int((labels == cluster_id).sum())
        # 설명: 같은 동작을 여러 번 반복해요.
        for cluster_id in sorted(np.unique(labels))
    # 설명: 다음 코드를 실행해요.
    }

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter32",
        # 설명: 다음 코드를 실행해요.
        "topic": "K-Means 맛보기",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "labels": labels.tolist(),
        # 설명: 다음 코드를 실행해요.
        "cluster_counts": cluster_counts,
        # 설명: 다음 코드를 실행해요.
        "centers": model.cluster_centers_.round(3).tolist(),
        # 설명: 다음 코드를 실행해요.
        "inertia": round(float(model.inertia_), 4),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
