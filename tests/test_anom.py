import numpy as np

from anom.detect import iforest_flags, make_data, precision_recall, zscore_flags


def test_precision_recall_math():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 1, 0])
    assert precision_recall(y, p) == (0.5, 0.5)
    assert precision_recall(y, np.zeros(4)) == (0.0, 0.0)


def test_data_labels():
    X, y = make_data()
    assert X.shape == (1030, 2) and int(y.sum()) == 30


def test_flags_binary_and_deterministic():
    X, _ = make_data()
    assert set(np.unique(iforest_flags(X))) <= {0, 1}
    assert (iforest_flags(X) == iforest_flags(X)).all()


def test_iforest_recall_beats_zscore_on_correlation_breaking_anomalies():
    X, y = make_data()
    _, r_z = precision_recall(y, zscore_flags(X))
    _, r_i = precision_recall(y, iforest_flags(X))
    assert r_i > r_z
