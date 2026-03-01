"""학습률의 영향 실습 파일"""
from __future__ import annotations


def _loss(x: float) -> float:
    return (x - 2.0) ** 2


def _grad(x: float) -> float:
    return 2.0 * (x - 2.0)


def _optimize(lr: float, steps: int = 25) -> dict:
    x = -6.0
    losses = []
    diverged = False

    for _ in range(steps):
        x -= lr * _grad(x)
        current_loss = _loss(x)
        losses.append(round(float(current_loss), 6))

        if abs(x) > 1e6 or current_loss > 1e12:
            diverged = True
            break

    return {
        "learning_rate": lr,
        "end_x": round(float(x), 6),
        "end_loss": round(float(_loss(x)), 6),
        "steps_ran": len(losses),
        "diverged": diverged,
        "first_5_losses": losses[:5],
        "last_5_losses": losses[-5:],
    }


def run() -> dict:
    results = [_optimize(0.02), _optimize(0.2), _optimize(1.1)]

    return {
        "chapter": "chapter73",
        "topic": "학습률의 영향",
        "lesson_10min": "학습률이 너무 작으면 느리고, 너무 크면 발산할 수 있다.",
        "practice_30min": "세 가지 learning rate를 비교해 수렴/발산을 확인한다.",
        "results": results,
    }


if __name__ == "__main__":
    print(run())
