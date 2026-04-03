"""Spatial validation primitive: certifies a trained model for inference.

This module implements the validation gate that train.py runs after fitting the
Random Forest model. A model is only written to disk if validate_spatial() returns
a ValidationResult whose AUC meets the SPATIAL_VALIDATION_THRESHOLD.

validate_spatial() is a pure function — no I/O, no global state. It takes arrays
of ground-truth labels and predicted probabilities (from RF.predict_proba) and
returns a fully-populated ValidationResult.

Metrics
-------
AUC (Area Under the ROC Curve)
    Computed via the trapezoidal rule over the ROC curve. Threshold-independent
    measure of discriminability. Primary gate metric: model is certified only if
    AUC >= SPATIAL_VALIDATION_THRESHOLD.

Precision, recall
    Computed at the threshold that maximises F1 (optimal operating point).
    Saved as diagnostics; not used as a gate.

Calibration error (ECE — Expected Calibration Error)
    Mean absolute difference between predicted probability and fraction of
    positives, binned into 10 equal-width bins. Values near 0 indicate
    well-calibrated probabilities; values near 0.5 indicate systematic
    over- or under-confidence.

Confusion matrix
    2×2 [[TN, FP], [FN, TP]] at the F1-optimal threshold. Stored as a tuple
    of four ints: (tn, fp, fn, tp).

Spatial validation
------------------
The name "spatial validation" refers to how the caller should split data before
calling this function. The caller (train.py) should hold out points from a
different geographic region than the training set so that spatial autocorrelation
does not inflate the AUC. This function itself is agnostic to the split strategy
— it receives labels and probabilities and returns metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Metrics from a single spatial validation run.

    All fields are populated by validate_spatial(). No field is optional.

    Attributes
    ----------
    auc:
        Area Under the ROC Curve in [0, 1]. AUC = 0.5 is random; AUC = 1.0
        is perfect discrimination.
    precision:
        Precision at the F1-optimal threshold: TP / (TP + FP).
        1.0 if no positive predictions are made (no false alarms).
    recall:
        Recall at the F1-optimal threshold: TP / (TP + FN).
        0.0 if no positives are predicted.
    calibration_error:
        Expected Calibration Error (ECE) in [0, 1]. Mean |predicted prob -
        fraction of positives| over 10 equal-width probability bins.
    confusion_matrix:
        (tn, fp, fn, tp) at the F1-optimal threshold.
    n_presence:
        Number of ground-truth positive (presence) samples.
    n_absence:
        Number of ground-truth negative (absence) samples.
    """

    auc: float
    precision: float
    recall: float
    calibration_error: float
    confusion_matrix: tuple[int, int, int, int]  # (tn, fp, fn, tp)
    n_presence: int
    n_absence: int

    def passes_gate(self, threshold: float) -> bool:
        """Return True if AUC meets the validation threshold."""
        return self.auc >= threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roc_auc(labels: list[int], probs: list[float]) -> float:
    """Compute AUC via the trapezoidal rule over the ROC curve.

    Uses the Mann-Whitney U statistic formulation:
        AUC = (number of (pos, neg) pairs where pos_prob > neg_prob
               + 0.5 * ties) / (n_pos * n_neg)

    This is numerically equivalent to the trapezoidal ROC AUC and avoids
    sorting the full threshold sweep.
    """
    pos_probs = [p for l, p in zip(labels, probs) if l == 1]
    neg_probs = [p for l, p in zip(labels, probs) if l == 0]

    n_pos = len(pos_probs)
    n_neg = len(neg_probs)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined — return chance level

    concordant = 0.0
    for pp in pos_probs:
        for np_ in neg_probs:
            if pp > np_:
                concordant += 1.0
            elif pp == np_:
                concordant += 0.5

    return concordant / (n_pos * n_neg)


def _f1_optimal_threshold(
    labels: list[int], probs: list[float]
) -> float:
    """Find the threshold in probs that maximises F1."""
    thresholds = sorted(set(probs), reverse=True)

    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        tp = sum(1 for l, p in zip(labels, probs) if l == 1 and p >= thresh)
        fp = sum(1 for l, p in zip(labels, probs) if l == 0 and p >= thresh)
        fn = sum(1 for l, p in zip(labels, probs) if l == 1 and p < thresh)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def _confusion_at_threshold(
    labels: list[int], probs: list[float], threshold: float
) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) at a given threshold."""
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    return (tn, fp, fn, tp)


def _calibration_error(
    labels: list[int], probs: list[float], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) over n_bins equal-width bins."""
    n = len(labels)
    if n == 0:
        return 0.0

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge in the last bin
        if i == n_bins - 1:
            indices = [j for j, p in enumerate(probs) if lo <= p <= hi]
        else:
            indices = [j for j, p in enumerate(probs) if lo <= p < hi]

        if not indices:
            continue

        bin_prob = sum(probs[j] for j in indices) / len(indices)
        bin_frac = sum(labels[j] for j in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_prob - bin_frac)

    return ece


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_spatial(
    labels: list[int],
    probabilities: list[float],
) -> ValidationResult:
    """Compute spatial validation metrics from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    labels:
        Ground-truth binary labels: 1 = presence, 0 = absence.
        Must have at least one presence and one absence sample.
    probabilities:
        Predicted probability of presence for each sample, in [0, 1].
        Typically RF.predict_proba(X)[:, 1].
        Must be the same length as labels.

    Returns
    -------
    ValidationResult
        Fully populated result with AUC, precision, recall, calibration_error,
        confusion_matrix, n_presence, n_absence.

    Raises
    ------
    ValueError
        If labels and probabilities have different lengths.
    ValueError
        If labels contains no positive samples or no negative samples
        (AUC is undefined in that case).
    ValueError
        If any probability is outside [0, 1].
    """
    if len(labels) != len(probabilities):
        raise ValueError(
            f"labels and probabilities must have the same length: "
            f"got {len(labels)} labels and {len(probabilities)} probabilities."
        )

    n_presence = sum(1 for l in labels if l == 1)
    n_absence = sum(1 for l in labels if l == 0)

    if n_presence == 0:
        raise ValueError(
            "labels contains no positive (presence) samples. "
            "AUC is undefined without both classes present."
        )
    if n_absence == 0:
        raise ValueError(
            "labels contains no negative (absence) samples. "
            "AUC is undefined without both classes present."
        )

    bad_probs = [p for p in probabilities if not (0.0 <= p <= 1.0)]
    if bad_probs:
        raise ValueError(
            f"probabilities must be in [0, 1]; found out-of-range values: "
            f"{bad_probs[:5]}{'...' if len(bad_probs) > 5 else ''}"
        )

    labels_list = list(labels)
    probs_list = list(probabilities)

    auc = _roc_auc(labels_list, probs_list)
    threshold = _f1_optimal_threshold(labels_list, probs_list)
    tn, fp, fn, tp = _confusion_at_threshold(labels_list, probs_list, threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cal_error = _calibration_error(labels_list, probs_list)

    return ValidationResult(
        auc=auc,
        precision=precision,
        recall=recall,
        calibration_error=cal_error,
        confusion_matrix=(tn, fp, fn, tp),
        n_presence=n_presence,
        n_absence=n_absence,
    )
