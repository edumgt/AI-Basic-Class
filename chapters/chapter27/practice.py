# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""학습률에 따른 경사하강법 비교"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations


# 설명: `optimize` 함수를 만들어요.
def optimize(lr: float, steps: int = 25) -> list[float]:
    # 설명: 값을 저장하거나 바꿔요.
    x = 8.0
    # 설명: 값을 저장하거나 바꿔요.
    history = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for _ in range(steps):
        # 설명: 값을 저장하거나 바꿔요.
        grad = 2 * (x - 3)
        # 설명: 값을 저장하거나 바꿔요.
        x -= lr * grad
        # 설명: 다음 코드를 실행해요.
        history.append(x)
    # 설명: 함수 결과를 돌려줘요.
    return history


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    h_small = optimize(lr=0.05)
    # 설명: 값을 저장하거나 바꿔요.
    h_good = optimize(lr=0.2)
    # 설명: 값을 저장하거나 바꿔요.
    h_big = optimize(lr=1.1)

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter27",
        # 설명: 다음 코드를 실행해요.
        "topic": "학습률 실험",
        # 설명: 다음 코드를 실행해요.
        "final_x_small_lr": round(h_small[-1], 6),
        # 설명: 다음 코드를 실행해요.
        "final_x_good_lr": round(h_good[-1], 6),
        # 설명: 다음 코드를 실행해요.
        "final_x_big_lr": round(h_big[-1], 6),
        # 설명: 다음 코드를 실행해요.
        "big_lr_first_5": [round(v, 4) for v in h_big[:5]],
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
