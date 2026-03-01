"""성장 회고와 다음 단계 실습 파일"""
from __future__ import annotations

from pathlib import Path


LESSON_10MIN = "회고는 배운 점을 구조화하고 다음 학습 목표를 명확히 만든다."
PRACTICE_30MIN = "포트폴리오 README 템플릿을 생성해 학습 결과를 정리한다."


def run() -> dict:
    template = """# AI/ML Learning Portfolio\n\n## 1. 내가 해결한 문제\n- 예: 학습 데이터로 분류 모델 만들기\n\n## 2. 사용한 기술\n- Python, pandas, scikit-learn, FastAPI\n\n## 3. 핵심 결과\n- 정확도/F1 등 지표\n\n## 4. 실패와 개선\n- 실패 사례와 다음 실험 계획\n\n## 5. 다음 단계\n- 딥러닝 실전 프로젝트 1개\n- 배포 자동화 1개\n"""

    out_path = Path(__file__).with_name("PORTFOLIO_TEMPLATE.md")
    out_path.write_text(template, encoding="utf-8")

    return {
        "chapter": "chapter99",
        "topic": "성장 회고와 다음 단계",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "generated_template": str(out_path),
        "next_goals": [
            "실데이터 기반 프로젝트 1개 완성",
            "모델 성능 개선 실험 3회 기록",
            "API + 프론트 연동 데모 1회 배포",
        ],
    }


if __name__ == "__main__":
    print(run())
