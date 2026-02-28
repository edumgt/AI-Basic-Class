"""가중치 행렬과 순전파 기초"""
from __future__ import annotations

import numpy as np


def run() -> dict:
    X = np.array(
        [
            [0.5, 1.0, 0.2],
            [1.5, 0.3, 0.7],
            [0.1, 0.4, 1.2],
        ]
    )
    W = np.array(
        [
            [0.2, -0.1],
            [0.4, 0.3],
            [-0.5, 0.2],
        ]
    )
    b = np.array([[0.1, -0.2]])

    z = X @ W + b

    return {
        "chapter": "chapter22",
        "topic": "가중치 행렬과 순전파",
        "input_shape": list(X.shape),
        "weight_shape": list(W.shape),
        "bias_shape": list(b.shape),
        "linear_output": np.round(z, 4).tolist(),
    }


if __name__ == "__main__":
    print(run())
