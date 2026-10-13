import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def concordance_index(y, prediction):
    """Exact O(N log N) CI. True-label ties excluded; prediction ties count 0.5.

    This is a regression ranking index, not a censored survival estimator.
    """
    y, prediction = np.asarray(y), np.asarray(prediction)
    ranks = np.searchsorted(np.unique(prediction), prediction) + 1
    tree = np.zeros(ranks.max() + 1, dtype=np.int64)

    def prefix(i):
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    order = np.argsort(y, kind="stable")
    concordant, comparable, seen, start = 0.0, 0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and y[order[end]] == y[order[start]]:
            end += 1
        group = ranks[order[start:end]]
        for rank in group:
            lower = prefix(rank - 1)
            equal = prefix(rank) - lower
            concordant += lower + 0.5 * equal
            comparable += seen
        for rank in group:
            i = rank
            while i < len(tree):
                tree[i] += 1
                i += i & -i
        seen += len(group)
        start = end
    return float(concordant / comparable) if comparable else None


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("Metrics require nonempty equal-sized 1D arrays")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite labels/predictions")
    correlated = len(y) > 1 and np.ptp(y) > 0 and np.ptp(prediction) > 0
    return {"n": len(y), "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "pearson": float(pearsonr(y, prediction).statistic) if correlated else None,
            "spearman": float(spearmanr(y, prediction).statistic) if correlated else None,
            "ci": concordance_index(y, prediction)}
