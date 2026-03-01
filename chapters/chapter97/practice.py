"""발표용 스토리 만들기 실습 파일"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


LESSON_10MIN = "발표는 문제-방법-결과-한계의 흐름으로 구성해야 전달력이 높다."
PRACTICE_30MIN = "슬라이드용 차트 3개를 생성해 스토리 구조를 만든다."


def run() -> dict:
    out_dir = Path(__file__).parent

    x = [1, 2, 3, 4, 5]
    y_problem = [50, 48, 47, 49, 50]
    y_result = [50, 55, 60, 64, 67]

    fig1 = plt.figure(figsize=(4, 3))
    plt.plot(x, y_problem)
    plt.title("Problem Trend")
    p1 = out_dir / "slide_01_problem.png"
    fig1.savefig(p1, dpi=120)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(4, 3))
    plt.bar(["baseline", "new"], [0.72, 0.84])
    plt.title("Method Comparison")
    p2 = out_dir / "slide_02_method.png"
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(4, 3))
    plt.plot(x, y_result)
    plt.title("Result Growth")
    p3 = out_dir / "slide_03_result.png"
    fig3.savefig(p3, dpi=120)
    plt.close(fig3)

    return {
        "chapter": "chapter97",
        "topic": "발표용 스토리 만들기",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "generated_files": [str(p1), str(p2), str(p3)],
        "story_order": ["문제", "방법", "결과", "한계"],
    }


if __name__ == "__main__":
    print(run())
