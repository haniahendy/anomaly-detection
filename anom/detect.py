import numpy as np
from sklearn.ensemble import IsolationForest


def make_data(n: int = 1000, n_anom: int = 30, seed: int = 4):
    """2-D correlated normal data plus injected anomalies. Returns X, y (1 = anomaly)."""
    rng = np.random.default_rng(seed)
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X = rng.multivariate_normal([0, 0], cov, n)
    # anomalies break the correlation, so single-feature z-scores do not catch them easily
    A = np.column_stack([rng.uniform(-2.5, 2.5, n_anom), rng.uniform(-2.5, 2.5, n_anom)])
    A[:, 1] = -A[:, 0] * 1.2 + rng.normal(0, 0.3, n_anom)
    y = np.r_[np.zeros(n), np.ones(n_anom)]
    return np.vstack([X, A]), y


def zscore_flags(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    z = np.abs((X - X.mean(0)) / X.std(0))
    return (z.max(axis=1) > threshold).astype(int)


def iforest_flags(X: np.ndarray, contamination: float = 0.03, seed: int = 0) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    return (model.predict(X) == -1).astype(int)


def precision_recall(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    p = tp / max(int(y_pred.sum()), 1)
    r = tp / max(int(y_true.sum()), 1)
    return round(p, 3), round(r, 3)
