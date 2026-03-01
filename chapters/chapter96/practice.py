"""개인정보와 보안 실습 파일"""
from __future__ import annotations

import pandas as pd


LESSON_10MIN = "민감정보는 최소 수집하고, 저장 전 마스킹/비식별화를 적용해야 한다."
PRACTICE_30MIN = "이메일과 전화번호를 마스킹해 안전한 출력 형태를 만든다."


def mask_email(email: str) -> str:
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[:2] + "*" * (len(local) - 2)
    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) < 7:
        return "***"
    return f"{digits[:3]}-****-{digits[-4:]}"


def run() -> dict:
    df = pd.DataFrame(
        {
            "name": ["Mina", "Joon"],
            "email": ["mina.choi@example.com", "joon.kim@example.com"],
            "phone": ["010-1234-5678", "010-9876-5432"],
            "score": [88, 92],
        }
    )

    masked = df.copy()
    masked["email"] = masked["email"].map(mask_email)
    masked["phone"] = masked["phone"].map(mask_phone)

    return {
        "chapter": "chapter96",
        "topic": "개인정보와 보안",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "original_preview": df.to_dict(orient="records"),
        "masked_preview": masked.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(run())
