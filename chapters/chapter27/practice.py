"""학습률에 따른 경사하강법 비교"""
from __future__ import annotations


def optimize(lr: float, steps: int = 25) -> list[float]:
    x = 8.0
    history = []
    for _ in range(steps):
        grad = 2 * (x - 3)
        x -= lr * grad
        history.append(x)
    return history


def run() -> dict:
    h_small = optimize(lr=0.05)
    h_good = optimize(lr=0.2)
    h_big = optimize(lr=1.1)

    return {
        "chapter": "chapter27",
        "topic": "학습률 실험",
        "final_x_small_lr": round(h_small[-1], 6),
        "final_x_good_lr": round(h_good[-1], 6),
        "final_x_big_lr": round(h_big[-1], 6),
        "big_lr_first_5": [round(v, 4) for v in h_big[:5]],
    }


if __name__ == "__main__":
    print(run())
