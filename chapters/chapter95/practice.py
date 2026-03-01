"""AI 윤리와 편향 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "모델 성능뿐 아니라 집단별 결과 차이(편향)도 함께 봐야 한다."
PRACTICE_30MIN = "성별 집단별 승인율을 비교해 간단한 편향 지표를 계산한다."


def run() -> dict:
    df = pd.DataFrame(
        {
            "gender": ["F", "F", "F", "F", "M", "M", "M", "M", "M", "F"],
            "pred_approve": [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        }
    )

    group_rate = df.groupby("gender")["pred_approve"].mean().to_dict()
    parity_gap = abs(float(group_rate.get("F", 0.0)) - float(group_rate.get("M", 0.0)))

    return {
        "chapter": "chapter95",
        "topic": "AI 윤리와 편향",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "approval_rate_by_gender": {k: round(float(v), 4) for k, v in group_rate.items()},
        "demographic_parity_gap": round(parity_gap, 4),
        "note": "gap이 클수록 집단 간 결과 차이가 크므로 추가 점검이 필요하다.",
    }


if __name__ == "__main__":
    print(run())
