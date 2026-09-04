"""
Evaluation metrics for HVNet.

This module implements the classification metrics reported in the
HVNet manuscript:

    1. Accuracy
    2. Macro-specificity
    3. Macro one-vs-rest AUC
    4. Cohen's kappa

The task contains three classes:

    Normal
    OAG
    ACG

Metric definitions
------------------

Accuracy:

    Accuracy =
        Number of correctly classified samples
        ----------------------------------------
              Total number of samples


Class-specific specificity:

    Specificity_k =
        TN_k
        -----
        TN_k + FP_k


Macro-specificity:

    Macro-Specificity =
        (1 / K) * sum_k Specificity_k


AUC:

    Multiclass macro one-vs-rest ROC AUC.


Cohen's kappa:

    Standard multiclass Cohen's kappa.


Scaling
-------

The manuscript reports all four metrics on a 0--100 scale.

For example:

    Accuracy = 0.9206

is returned as:

    92.06

when scale=100.0.

Set scale=1.0 if decimal values in the range [0, 1] are preferred.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


# ======================================================================
# Type utilities
# ======================================================================


ArrayLike = Union[
    np.ndarray,
    torch.Tensor,
    Sequence[int],
    Sequence[float],
]


def _to_numpy(
    x: ArrayLike,
) -> np.ndarray:
    """
    Convert torch tensors or Python sequences to NumPy arrays.
    """

    if isinstance(x, torch.Tensor):
        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


# ======================================================================
# Input validation
# ======================================================================


def _prepare_targets(
    y_true: ArrayLike,
) -> np.ndarray:
    """
    Prepare integer ground-truth labels.

    Accepted shapes:

        [N]
        [N, 1]
    """

    y_true = _to_numpy(
        y_true
    )

    if (
        y_true.ndim == 2
        and y_true.shape[1] == 1
    ):
        y_true = y_true.squeeze(1)

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must have shape [N] or [N, 1], "
            f"but received {y_true.shape}."
        )

    return y_true.astype(
        np.int64
    )


def _prepare_probabilities(
    y_prob: ArrayLike,
    num_classes: int,
) -> np.ndarray:
    """
    Prepare multiclass predictive probabilities.

    Expected shape:

        [N, K]
    """

    y_prob = _to_numpy(
        y_prob
    )

    if y_prob.ndim != 2:
        raise ValueError(
            "y_prob must have shape [N, K], "
            f"but received {y_prob.shape}."
        )

    if y_prob.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} probability columns, "
            f"but received {y_prob.shape[1]}."
        )

    if not np.isfinite(y_prob).all():
        raise ValueError(
            "y_prob contains NaN or infinite values."
        )

    return y_prob.astype(
        np.float64
    )


# ======================================================================
# Specificity
# ======================================================================


def specificity_per_class(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Compute one-vs-rest specificity for each class.

    For class k:

        Specificity_k =
            TN_k / (TN_k + FP_k)

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
        Shape: [N].

    y_pred : array-like
        Predicted class labels.
        Shape: [N].

    num_classes : int
        Number of classes.

        HVNet:
            3

    Returns
    -------
    np.ndarray
        Per-class specificity.
        Shape: [K].
    """

    y_true = _prepare_targets(
        y_true
    )

    y_pred = _prepare_targets(
        y_pred
    )

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain "
            "the same number of samples."
        )

    labels = np.arange(
        num_classes
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    specificities = []

    total = cm.sum()

    for class_index in range(
        num_classes
    ):

        tp = cm[
            class_index,
            class_index,
        ]

        fn = (
            cm[
                class_index,
                :
            ].sum()
            - tp
        )

        fp = (
            cm[
                :,
                class_index
            ].sum()
            - tp
        )

        tn = (
            total
            - tp
            - fn
            - fp
        )

        denominator = (
            tn
            + fp
        )

        if denominator == 0:
            specificity = np.nan

        else:
            specificity = (
                tn
                / denominator
            )

        specificities.append(
            specificity
        )

    return np.asarray(
        specificities,
        dtype=np.float64,
    )


def macro_specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: int = 3,
) -> float:
    """
    Compute macro one-vs-rest specificity.

    Implements:

        Macro-Specificity =
            (1 / K)
            sum_k Specificity_k
    """

    specificities = specificity_per_class(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    if np.isnan(
        specificities
    ).any():
        raise ValueError(
            "Specificity is undefined for at least one class. "
            "Check whether the evaluation cohort contains "
            "sufficient negative samples for every class."
        )

    return float(
        specificities.mean()
    )


# ======================================================================
# Multiclass AUC
# ======================================================================


def macro_ovr_auc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    num_classes: int = 3,
) -> float:
    """
    Compute multiclass macro one-vs-rest ROC AUC.

    Implements the manuscript evaluation protocol:

        average = "macro"
        multi_class = "ovr"

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
        Shape: [N].

    y_prob : array-like
        Predictive class probabilities.
        Shape: [N, K].

    num_classes : int
        Number of classes.

    Returns
    -------
    float
        Macro OvR AUC in the range [0, 1].
    """

    y_true = _prepare_targets(
        y_true
    )

    y_prob = _prepare_probabilities(
        y_prob,
        num_classes=num_classes,
    )

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            "y_true and y_prob must contain "
            "the same number of samples."
        )

    unique_classes = np.unique(
        y_true
    )

    expected_classes = np.arange(
        num_classes
    )

    if not np.array_equal(
        unique_classes,
        expected_classes,
    ):
        raise ValueError(
            "Macro multiclass AUC requires all classes to be "
            "represented in the evaluation cohort. "
            f"Expected classes {expected_classes.tolist()}, "
            f"but observed {unique_classes.tolist()}."
        )

    auc = roc_auc_score(
        y_true,
        y_prob,
        labels=expected_classes,
        multi_class="ovr",
        average="macro",
    )

    return float(
        auc
    )


# ======================================================================
# Complete HVNet metric computation
# ======================================================================


def compute_hvnet_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    num_classes: int = 3,
    scale: float = 100.0,
    return_per_class: bool = False,
    return_confusion_matrix: bool = False,
) -> Dict[str, Union[
    float,
    np.ndarray,
]]:
    """
    Compute all evaluation metrics reported in the HVNet manuscript.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
        Shape: [N].

    y_prob : array-like
        Model predictive probabilities.
        Shape: [N, K].

    num_classes : int
        Number of classes.

        HVNet:
            3

    scale : float
        Metric scaling factor.

        Manuscript:
            100.0

        Use:
            1.0

        for conventional [0, 1] values.

    return_per_class : bool
        If True, return per-class specificity.

    return_confusion_matrix : bool
        If True, return the confusion matrix.

    Returns
    -------
    dict

        accuracy
            Three-class accuracy.

        specificity
            Macro one-vs-rest specificity.

        auc
            Macro one-vs-rest AUC.

        kappa
            Multiclass Cohen's kappa.

        Optionally:

        specificity_per_class
        confusion_matrix
        predictions
    """

    if scale <= 0:
        raise ValueError(
            "scale must be greater than zero."
        )

    y_true = _prepare_targets(
        y_true
    )

    y_prob = _prepare_probabilities(
        y_prob,
        num_classes=num_classes,
    )

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            "y_true and y_prob must contain "
            "the same number of samples."
        )

    if y_true.size == 0:
        raise ValueError(
            "Cannot compute metrics on an empty cohort."
        )

    # --------------------------------------------------------------
    # Predicted class
    #
    # argmax_k p_final,k
    # --------------------------------------------------------------

    y_pred = np.argmax(
        y_prob,
        axis=1,
    )

    # ==============================================================
    # Accuracy
    # ==============================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # ==============================================================
    # Macro-specificity
    # ==============================================================

    per_class_spec = specificity_per_class(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    if np.isnan(
        per_class_spec
    ).any():
        raise ValueError(
            "Specificity is undefined for at least one class."
        )

    specificity = per_class_spec.mean()

    # ==============================================================
    # Macro one-vs-rest AUC
    # ==============================================================

    auc = macro_ovr_auc(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=num_classes,
    )

    # ==============================================================
    # Cohen's kappa
    # ==============================================================

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=np.arange(
            num_classes
        ),
    )

    # ==============================================================
    # Output
    # ==============================================================

    results = {
        "accuracy":
            float(accuracy * scale),

        "specificity":
            float(specificity * scale),

        "auc":
            float(auc * scale),

        "kappa":
            float(kappa * scale),

        "predictions":
            y_pred,
    }

    if return_per_class:

        results[
            "specificity_per_class"
        ] = (
            per_class_spec
            * scale
        )

    if return_confusion_matrix:

        results[
            "confusion_matrix"
        ] = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(
                num_classes
            ),
        )

    return results


# ======================================================================
# Batch-wise metric accumulator
# ======================================================================


class HVNetMetricTracker:
    """
    Accumulate predictions across batches and compute cohort-level metrics.

    AUC and specificity should generally NOT be averaged independently
    across mini-batches.

    Instead:

        1. Accumulate all labels and probabilities;
        2. Concatenate them;
        3. Compute metrics once for the complete validation/test cohort.

    This class implements that workflow.

    Example
    -------

    tracker = HVNetMetricTracker()

    for batch in loader:

        outputs = model(...)

        tracker.update(
            targets=batch["label"],
            probabilities=outputs["p_final"],
        )

    metrics = tracker.compute()
    """

    def __init__(
        self,
        num_classes: int = 3,
        scale: float = 100.0,
    ) -> None:

        self.num_classes = num_classes
        self.scale = scale

        self.reset()

    def reset(
        self,
    ) -> None:
        """
        Clear all accumulated samples.
        """

        self._targets: List[np.ndarray] = []
        self._probabilities: List[np.ndarray] = []

    def update(
        self,
        targets: ArrayLike,
        probabilities: ArrayLike,
    ) -> None:
        """
        Add one batch of labels and predictions.
        """

        targets = _prepare_targets(
            targets
        )

        probabilities = _prepare_probabilities(
            probabilities,
            num_classes=self.num_classes,
        )

        if targets.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "targets and probabilities must have "
                "the same batch size."
            )

        self._targets.append(
            targets
        )

        self._probabilities.append(
            probabilities
        )

    def compute(
        self,
        return_per_class: bool = False,
        return_confusion_matrix: bool = False,
    ) -> Dict[str, Union[
        float,
        np.ndarray,
    ]]:
        """
        Compute metrics over all accumulated samples.
        """

        if len(
            self._targets
        ) == 0:
            raise RuntimeError(
                "No predictions have been accumulated."
            )

        y_true = np.concatenate(
            self._targets,
            axis=0,
        )

        y_prob = np.concatenate(
            self._probabilities,
            axis=0,
        )

        return compute_hvnet_metrics(
            y_true=y_true,
            y_prob=y_prob,
            num_classes=self.num_classes,
            scale=self.scale,
            return_per_class=return_per_class,
            return_confusion_matrix=
                return_confusion_matrix,
        )

    def __len__(
        self,
    ) -> int:
        """
        Number of accumulated samples.
        """

        return sum(
            len(x)
            for x in self._targets
        )


# ======================================================================
# Five-fold result aggregation
# ======================================================================


def summarize_fold_metrics(
    fold_metrics: Sequence[Dict[str, float]],
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple fold-specific models.

    This matches the manuscript reporting strategy:

        mean +/- standard deviation

    across the five fold-specific models.

    Parameters
    ----------
    fold_metrics : sequence of dict
        Example:

        [
            {
                "accuracy": 91.8,
                "specificity": 95.6,
                "auc": 97.9,
                "kappa": 87.7,
            },
            ...
        ]

    metric_names : sequence of str, optional
        Metrics to summarize.

        Default:

            accuracy
            specificity
            auc
            kappa

    Returns
    -------
    dict

        {
            "accuracy": {
                "mean": ...,
                "std": ...
            },

            ...
        }

    Notes
    -----
    Standard deviation uses:

        ddof = 1

    corresponding to sample standard deviation across fold-specific
    model results.
    """

    if len(
        fold_metrics
    ) == 0:
        raise ValueError(
            "fold_metrics must not be empty."
        )

    if metric_names is None:

        metric_names = [
            "accuracy",
            "specificity",
            "auc",
            "kappa",
        ]

    summary = {}

    for metric_name in metric_names:

        values = []

        for fold_index, metrics in enumerate(
            fold_metrics
        ):

            if metric_name not in metrics:
                raise KeyError(
                    f"Metric '{metric_name}' is missing from "
                    f"fold {fold_index}."
                )

            values.append(
                float(
                    metrics[
                        metric_name
                    ]
                )
            )

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        # With five fold-specific models the manuscript reports
        # mean +/- SD.
        #
        # For a single result, SD is defined here as 0 for convenience.
        if len(values) > 1:

            std = values.std(
                ddof=1
            )

        else:

            std = 0.0

        summary[
            metric_name
        ] = {
            "mean":
                float(
                    values.mean()
                ),

            "std":
                float(
                    std
                ),
        }

    return summary


# ======================================================================
# Formatting utility
# ======================================================================


def format_mean_std(
    mean: float,
    std: float,
    decimals: int = 2,
) -> str:
    """
    Format a result in manuscript/table style.

    Example:

        92.06 +/- 0.42
    """

    return (
        f"{mean:.{decimals}f} "
        f"± "
        f"{std:.{decimals}f}"
    )


# ======================================================================
# Minimal sanity check
# ======================================================================


if __name__ == "__main__":

    # --------------------------------------------------------------
    # Artificial three-class example.
    # No clinical data are used.
    # --------------------------------------------------------------

    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            2,
            2,
        ]
    )

    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.70, 0.20, 0.10],

            [0.10, 0.80, 0.10],
            [0.20, 0.60, 0.20],

            [0.05, 0.10, 0.85],
            [0.10, 0.20, 0.70],
        ]
    )

    metrics = compute_hvnet_metrics(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=3,
        scale=100.0,
        return_per_class=True,
        return_confusion_matrix=True,
    )

    print(
        "Accuracy:",
        metrics["accuracy"],
    )

    print(
        "Macro-specificity:",
        metrics["specificity"],
    )

    print(
        "Macro OvR AUC:",
        metrics["auc"],
    )

    print(
        "Cohen's kappa:",
        metrics["kappa"],
    )

    print(
        "Per-class specificity:",
        metrics["specificity_per_class"],
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )

    # --------------------------------------------------------------
    # MetricTracker example
    # --------------------------------------------------------------

    tracker = HVNetMetricTracker(
        num_classes=3,
        scale=100.0,
    )

    tracker.update(
        targets=y_true[:3],
        probabilities=y_prob[:3],
    )

    tracker.update(
        targets=y_true[3:],
        probabilities=y_prob[3:],
    )

    tracked_metrics = tracker.compute()

    print(
        "\nTracker accuracy:",
        tracked_metrics["accuracy"],
    )

    # --------------------------------------------------------------
    # Five-fold summary example
    # --------------------------------------------------------------

    example_folds = [
        {
            "accuracy": 91.60,
            "specificity": 95.50,
            "auc": 97.90,
            "kappa": 87.40,
        },
        {
            "accuracy": 92.10,
            "specificity": 95.80,
            "auc": 98.00,
            "kappa": 88.10,
        },
        {
            "accuracy": 92.30,
            "specificity": 96.00,
            "auc": 98.20,
            "kappa": 88.50,
        },
        {
            "accuracy": 91.90,
            "specificity": 95.70,
            "auc": 98.10,
            "kappa": 87.80,
        },
        {
            "accuracy": 92.40,
            "specificity": 96.10,
            "auc": 98.00,
            "kappa": 88.40,
        },
    ]

    summary = summarize_fold_metrics(
        example_folds
    )

    print(
        "\nFive-fold summary:"
    )

    for metric_name, statistics in summary.items():

        print(
            metric_name,
            format_mean_std(
                statistics["mean"],
                statistics["std"],
            ),
        )
