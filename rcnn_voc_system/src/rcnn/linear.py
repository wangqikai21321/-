from __future__ import annotations

import numpy as np


class LinearSVM:
    """Small NumPy linear SVM with squared hinge loss and L2 regularization."""

    def __init__(self, c: float = 0.001, lr: float = 0.01, epochs: int = 20, batch_size: int = 256, seed: int = 42):
        self.c = c
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        y = np.where(y > 0, 1.0, -1.0)
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(x.shape[1], dtype=np.float32)
        self.b = 0.0
        for _ in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                yb = y[idx]
                margins = yb * (xb @ self.w + self.b)
                active = margins < 1.0
                grad_w = self.w.copy()
                grad_b = 0.0
                if np.any(active):
                    miss = 1.0 - margins[active]
                    grad_w += self.c * (-2.0 * (miss * yb[active]) @ xb[active])
                    grad_b += float(self.c * np.sum(-2.0 * miss * yb[active]))
                self.w -= self.lr * grad_w / max(1, len(idx))
                self.b -= self.lr * grad_b / max(1, len(idx))
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("LinearSVM is not fitted.")
        return np.asarray(x, dtype=np.float32) @ self.w + self.b


class RidgeRegressor:
    def __init__(self, alpha: float = 1000.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
        eye = np.eye(xb.shape[1], dtype=np.float32)
        eye[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(xb.T @ xb + self.alpha * eye, xb.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegressor is not fitted.")
        x = np.asarray(x, dtype=np.float32)
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
        return xb @ self.coef_
