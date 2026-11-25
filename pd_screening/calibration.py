"""Normal-only calibration and reporting metrics."""

from fractions import Fraction
import numpy as np


def threshold(scores, alpha=0.05):
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite 1-D normal calibration array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    # Decimal arithmetic avoids float-ceil errors at integer rank boundaries.
    rank = (len(values)+1) * (1-Fraction(str(alpha)))
    k = (rank.numerator+rank.denominator-1)//rank.denominator
    return float(values[k-1]) if k <= len(values) else float("inf")


def metrics(normal_scores, event_scores, cut):
    fpr = float(np.mean(np.asarray(normal_scores) > cut))
    retention = float(np.mean(np.asarray(event_scores) > cut))
    return dict(fpr=fpr, retention=retention, ba=(retention+1-fpr)/2)
