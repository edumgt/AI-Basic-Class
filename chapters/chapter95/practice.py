# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""AI 윤리와 편향 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "모델 성능뿐 아니라 집단별 결과 차이(편향)도 함께 봐야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "성별 집단별 승인율을 비교해 간단한 편향 지표를 계산한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "gender": ["F", "F", "F", "F", "M", "M", "M", "M", "M", "F"],
            # 설명: 다음 코드를 실행해요.
            "pred_approve": [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    group_rate = df.groupby("gender")["pred_approve"].mean().to_dict()
    # 설명: 값을 저장하거나 바꿔요.
    parity_gap = abs(float(group_rate.get("F", 0.0)) - float(group_rate.get("M", 0.0)))

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter95",
        # 설명: 다음 코드를 실행해요.
        "topic": "AI 윤리와 편향",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "approval_rate_by_gender": {k: round(float(v), 4) for k, v in group_rate.items()},
        # 설명: 다음 코드를 실행해요.
        "demographic_parity_gap": round(parity_gap, 4),
        # 설명: 다음 코드를 실행해요.
        "note": "gap이 클수록 집단 간 결과 차이가 크므로 추가 점검이 필요하다.",
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
