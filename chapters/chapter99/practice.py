# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""성장 회고와 다음 단계 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from pathlib import Path


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "회고는 배운 점을 구조화하고 다음 학습 목표를 명확히 만든다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "포트폴리오 README 템플릿을 생성해 학습 결과를 정리한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    template = """# AI/ML Learning Portfolio\n\n## 1. 내가 해결한 문제\n- 예: 학습 데이터로 분류 모델 만들기\n\n## 2. 사용한 기술\n- Python, pandas, scikit-learn, FastAPI\n\n## 3. 핵심 결과\n- 정확도/F1 등 지표\n\n## 4. 실패와 개선\n- 실패 사례와 다음 실험 계획\n\n## 5. 다음 단계\n- 딥러닝 실전 프로젝트 1개\n- 배포 자동화 1개\n"""

    # 설명: 값을 저장하거나 바꿔요.
    out_path = Path(__file__).with_name("PORTFOLIO_TEMPLATE.md")
    # 설명: 값을 저장하거나 바꿔요.
    out_path.write_text(template, encoding="utf-8")

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter99",
        # 설명: 다음 코드를 실행해요.
        "topic": "성장 회고와 다음 단계",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "generated_template": str(out_path),
        # 설명: 다음 코드를 실행해요.
        "next_goals": [
            # 설명: 다음 코드를 실행해요.
            "실데이터 기반 프로젝트 1개 완성",
            # 설명: 다음 코드를 실행해요.
            "모델 성능 개선 실험 3회 기록",
            # 설명: 다음 코드를 실행해요.
            "API + 프론트 연동 데모 1회 배포",
        # 설명: 다음 코드를 실행해요.
        ],
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
