import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plcc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)

    unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inverse == i)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


def srcc(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    r1 = rankdata(y_true)
    r2 = rankdata(y_pred)
    if np.std(r1) < 1e-12 or np.std(r2) < 1e-12:
        return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])
