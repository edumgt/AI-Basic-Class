"""배치, 에폭, 반복 실습 파일"""
from __future__ import annotations

import math


LESSON_10MIN = "배치 크기와 에폭 수로 총 업데이트 횟수가 결정된다."
PRACTICE_30MIN = "batch size를 바꿔 step 수와 학습 노이즈를 비교한다."


def run() -> dict:
    dataset_size = 240
    epochs = 5
    batch_sizes = [1, 8, 32, 64]

    table = []
    for bsz in batch_sizes:
        steps_per_epoch = math.ceil(dataset_size / bsz)
        total_updates = steps_per_epoch * epochs
        relative_noise = round((1.0 / (bsz**0.5)), 4)

        table.append(
            {
                "batch_size": bsz,
                "steps_per_epoch": steps_per_epoch,
                "total_updates": total_updates,
                "relative_gradient_noise": relative_noise,
            }
        )

    return {
        "chapter": "chapter75",
        "topic": "배치, 에폭, 반복",
        "lesson_10min": LESSON_10MIN,
        "practice_30min": PRACTICE_30MIN,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "comparison": table,
    }


if __name__ == "__main__":
    print(run())
