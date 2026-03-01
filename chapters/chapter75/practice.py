# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""배치, 에폭, 반복 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
import math


# 설명: 값을 저장하거나 바꿔요.
LESSON_10MIN = "배치 크기와 에폭 수로 총 업데이트 횟수가 결정된다."
# 설명: 값을 저장하거나 바꿔요.
PRACTICE_30MIN = "batch size를 바꿔 step 수와 학습 노이즈를 비교한다."


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 값을 저장하거나 바꿔요.
    dataset_size = 240
    # 설명: 값을 저장하거나 바꿔요.
    epochs = 5
    # 설명: 값을 저장하거나 바꿔요.
    batch_sizes = [1, 8, 32, 64]

    # 설명: 값을 저장하거나 바꿔요.
    table = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for bsz in batch_sizes:
        # 설명: 값을 저장하거나 바꿔요.
        steps_per_epoch = math.ceil(dataset_size / bsz)
        # 설명: 값을 저장하거나 바꿔요.
        total_updates = steps_per_epoch * epochs
        # 설명: 값을 저장하거나 바꿔요.
        relative_noise = round((1.0 / (bsz**0.5)), 4)

        # 설명: 다음 코드를 실행해요.
        table.append(
            # 설명: 다음 코드를 실행해요.
            {
                # 설명: 다음 코드를 실행해요.
                "batch_size": bsz,
                # 설명: 다음 코드를 실행해요.
                "steps_per_epoch": steps_per_epoch,
                # 설명: 다음 코드를 실행해요.
                "total_updates": total_updates,
                # 설명: 다음 코드를 실행해요.
                "relative_gradient_noise": relative_noise,
            # 설명: 다음 코드를 실행해요.
            }
        # 설명: 다음 코드를 실행해요.
        )

    # 설명: 함수 결과를 돌려줘요.
    return {
        # 설명: 다음 코드를 실행해요.
        "chapter": "chapter75",
        # 설명: 다음 코드를 실행해요.
        "topic": "배치, 에폭, 반복",
        # 설명: 다음 코드를 실행해요.
        "lesson_10min": LESSON_10MIN,
        # 설명: 다음 코드를 실행해요.
        "practice_30min": PRACTICE_30MIN,
        # 설명: 다음 코드를 실행해요.
        "dataset_size": dataset_size,
        # 설명: 다음 코드를 실행해요.
        "epochs": epochs,
        # 설명: 다음 코드를 실행해요.
        "comparison": table,
    # 설명: 다음 코드를 실행해요.
    }


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
