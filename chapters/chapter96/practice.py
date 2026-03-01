# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""개인정보와 보안 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import pandas as pd


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "민감정보는 최소 수집하고, 저장 전 마스킹/비식별화를 적용해야 한다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "이메일과 전화번호를 마스킹해 안전한 출력 형태를 만든다."


# 설명: `mask_email` 함수를 만들어요.
def mask_email(email: str) -> str:
    # 설명: 조건이 맞는지 확인해요.
    if "@" not in email:
        # 설명: 함수 결과를 돌려줘요.
        return "***"
    # 설명: 값을 저장하거나 바꿔요.
    local, domain = email.split("@", 1)
    # 설명: 조건이 맞는지 확인해요.
    if len(local) <= 2:
        # 설명: 값을 저장하거나 바꿔요.
        masked_local = local[0] + "*"
    # 설명: 조건이 모두 아니면 이 부분을 실행해요.
    else:
        # 설명: 값을 저장하거나 바꿔요.
        masked_local = local[:2] + "*" * (len(local) - 2)
    # 설명: 함수 결과를 돌려줘요.
    return f"{masked_local}@{domain}"


# 설명: `mask_phone` 함수를 만들어요.
def mask_phone(phone: str) -> str:
    # 설명: 값을 저장하거나 바꿔요.
    digits = "".join(ch for ch in phone if ch.isdigit())
    # 설명: 조건이 맞는지 확인해요.
    if len(digits) < 7:
        # 설명: 함수 결과를 돌려줘요.
        return "***"
    # 설명: 함수 결과를 돌려줘요.
    return f"{digits[:3]}-****-{digits[-4:]}"


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    df = pd.DataFrame(
        # 설명: 다음 코드를 실행해요.
        {
            # 설명: 다음 코드를 실행해요.
            "name": ["Mina", "Joon"],
            # 설명: 다음 코드를 실행해요.
            "email": ["mina.choi@example.com", "joon.kim@example.com"],
            # 설명: 다음 코드를 실행해요.
            "phone": ["010-1234-5678", "010-9876-5432"],
            # 설명: 다음 코드를 실행해요.
            "score": [88, 92],
        # 설명: 다음 코드를 실행해요.
        }
    # 설명: 다음 코드를 실행해요.
    )

    # 설명: 값을 저장하거나 바꿔요.
    masked = df.copy()
    # 설명: 값을 저장하거나 바꿔요.
    masked["email"] = masked["email"].map(mask_email)
    # 설명: 값을 저장하거나 바꿔요.
    masked["phone"] = masked["phone"].map(mask_phone)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter96",
        # 설명: 다음 코드를 실행해요.
        "topic": "개인정보와 보안",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 값을 저장하거나 바꿔요.
        "original_preview": df.to_dict(orient="records"),
        # 설명: 값을 저장하거나 바꿔요.
        "masked_preview": masked.to_dict(orient="records"),
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
