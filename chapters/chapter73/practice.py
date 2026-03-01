# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""학습률의 영향 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: `_loss` 함수를 만들어요.
def _loss(x: float) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return (x - 2.0) ** 2


# 설명: `_grad` 함수를 만들어요.
def _grad(x: float) -> float:
    # 설명: 함수 결과를 돌려줘요.
    return 2.0 * (x - 2.0)


# 설명: `_optimize` 함수를 만들어요.
def _optimize(lr: float, steps: int = 25) -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    x = -6.0
    # 설명: 값을 저장하거나 바꿔요.
    losses = []
    # 설명: 값을 저장하거나 바꿔요.
    diverged = False

    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(steps):
        # 설명: 값을 저장하거나 바꿔요.
        x -= lr * _grad(x)
        # 설명: 값을 저장하거나 바꿔요.
        current_loss = _loss(x)
        # 설명: 다음 코드를 실행해요.
        losses.append(round(float(current_loss), 6))

        # 설명: 조건이 맞는지 확인해요.
        if abs(x) > 1e6 or current_loss > 1e12:
            # 설명: 값을 저장하거나 바꿔요.
            diverged = True
            # 설명: 다음 코드를 실행해요.
            break

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "learning_rate": lr,
        # 설명: 다음 코드를 실행해요.
        "end_x": round(float(x), 6),
        # 설명: 다음 코드를 실행해요.
        "end_loss": round(float(_loss(x)), 6),
        # 설명: 다음 코드를 실행해요.
        "steps_ran": len(losses),
        # 설명: 다음 코드를 실행해요.
        "diverged": diverged,
        # 설명: 다음 코드를 실행해요.
        "first_5_losses": losses[:5],
        # 설명: 다음 코드를 실행해요.
        "last_5_losses": losses[-5:],
    # 설명: 다음 코드를 실행해요.
    }


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    results = [_optimize(0.02), _optimize(0.2), _optimize(1.1)]

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter73",
        # 설명: 다음 코드를 실행해요.
        "topic": "학습률의 영향",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": "학습률이 너무 작으면 느리고, 너무 크면 발산할 수 있다.",
        # 설명: 다음 코드를 실행해요.
        "practice_30min": "세 가지 learning rate를 비교해 수렴/발산을 확인한다.",
        # 설명: 다음 코드를 실행해요.
        "results": results,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
