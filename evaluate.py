"""
Evaluation script for HVNet.

This script implements the evaluation protocol described in the
HVNet manuscript:

    Glaucoma Subtyping via Hypothesis-Guided Verification
    and Arbitrated Fusion

=======================================================================
Evaluation protocol
=======================================================================

Five fold-specific models are obtained from five-fold cross-validation
performed exclusively within the internal development subset.

Each selected fold-specific checkpoint is independently evaluated on:

    1. The same fixed internal test cohort;
    2. The same held-out community-referral cohort.

Neither evaluation cohort is used for:

    - model training;
    - validation;
    - hyperparameter selection;
    - checkpoint selection.

For each cohort, performance is summarized across the five
fold-specific models as:

    mean +/- standard deviation

for:

    - Accuracy
    - Macro-Specificity
    - Macro one-vs-rest AUC
    - Cohen's Kappa

All metrics are reported on a 0--100 scale, consistent with the
manuscript tables.

=======================================================================
Expected checkpoints
=======================================================================

By default:

    checkpoints/
    ├── fold_1/
    │   └── hvnet_fold1_best.pth
    ├── fold_2/
    │   └── hvnet_fold2_best.pth
    ├── fold_3/
    │   └── hvnet_fold3_best.pth
    ├── fold_4/
    │   └── hvnet_fold4_best.pth
    └── fold_5/
        └── hvnet_fold5_best.pth

=======================================================================
Important
=======================================================================

The fixed internal-test assignment must already exist in the metadata.

Unlike train.py, this script does NOT generate a fallback internal-test
split. This prevents accidental evaluation on a newly generated test set
that differs from the manuscript cohort.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader, Subset

from datasets.dataset_template import HVNetDataset
from models.hvnet import HVNet
from utils.metrics import (
    HVNetMetricTracker,
    format_mean_std,
    summarize_fold_metrics,
)
from utils.transforms import build_eval_transform


# ======================================================================
# Configuration
# ======================================================================


def load_config(
    config_path: str,
) -> Dict:
    """
    Load YAML configuration.
    """

    config_path = Path(
        config_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    return config


# ======================================================================
# Reproducibility
# ======================================================================


def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for evaluation.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


# ======================================================================
# Model construction
# ======================================================================


def build_model_from_config(
    config: Dict,
) -> HVNet:
    """
    Construct HVNet from a configuration dictionary.

    The model structure should match the structure used during training.
    Parameters that are not explicitly provided in the YAML use the
    defaults defined in models/hvnet.py.

    Before final public release, implementation-level parameters should
    be checked against the original experimental implementation.
    """

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    model_cfg = config.get(
        "model",
        {},
    )

    backbone_cfg = model_cfg.get(
        "backbone",
        {},
    )

    clinical_cfg = model_cfg.get(
        "clinical_encoder",
        {},
    )

    visual_cfg = model_cfg.get(
        "visual_verification",
        {},
    )

    ds_cfg = model_cfg.get(
        "ds_fusion",
        {},
    )

    gate_cfg = model_cfg.get(
        "confidence_gate",
        {},
    )

    backbone_out_indices = tuple(
        backbone_cfg.get(
            "out_indices",
            [1, 2],
        )
    )

    local_branch_cfg = visual_cfg.get(
        "local_branch",
        {},
    )

    model = HVNet(
        # ----------------------------------------------------------
        # Task
        # ----------------------------------------------------------

        num_classes=experiment_cfg.get(
            "num_classes",
            3,
        ),

        # ----------------------------------------------------------
        # Backbone
        # ----------------------------------------------------------

        backbone_name=backbone_cfg.get(
            "name",
            "convnext_tiny",
        ),

        # Pretrained weights are unnecessary when loading a complete
        # trained checkpoint. Setting pretrained=False also avoids
        # an unnecessary ImageNet download during evaluation.
        pretrained=False,

        backbone_out_indices=
            backbone_out_indices,

        # ----------------------------------------------------------
        # Clinical representation
        # ----------------------------------------------------------

        clinical_feature_dim=
            clinical_cfg.get(
                "clinical_feature_dim",
                256,
            ),

        clinical_embedding_dim=
            clinical_cfg.get(
                "embedding_dim",
                64,
            ),

        clinical_num_heads=
            clinical_cfg.get(
                "num_heads",
                4,
            ),

        clinical_num_layers=
            clinical_cfg.get(
                "num_layers",
                2,
            ),

        clinical_feedforward_dim=
            clinical_cfg.get(
                "feedforward_dim",
                128,
            ),

        clinical_dropout=
            clinical_cfg.get(
                "dropout",
                0.1,
            ),

        # ----------------------------------------------------------
        # MGVA
        # ----------------------------------------------------------

        attention_dim=
            visual_cfg.get(
                "attention_dim",
                256,
            ),

        global_dim=
            visual_cfg.get(
                "global_dim",
                256,
            ),

        local_dim=
            visual_cfg.get(
                "local_dim",
                256,
            ),

        visual_feature_dim=
            visual_cfg.get(
                "visual_feature_dim",
                256,
            ),

        visual_attention_heads=
            visual_cfg.get(
                "num_attention_heads",
                4,
            ),

        num_local_patches=
            local_branch_cfg.get(
                "num_local_patches",
                4,
            ),

        patch_size=
            local_branch_cfg.get(
                "patch_size",
                3,
            ),

        local_transformer_heads=
            local_branch_cfg.get(
                "transformer_heads",
                4,
            ),

        local_transformer_layers=
            local_branch_cfg.get(
                "transformer_layers",
                1,
            ),

        local_feedforward_dim=
            local_branch_cfg.get(
                "feedforward_dim",
                512,
            ),

        visual_dropout=
            visual_cfg.get(
                "dropout",
                0.1,
            ),

        # ----------------------------------------------------------
        # DS fusion
        # ----------------------------------------------------------

        ds_temperature=
            ds_cfg.get(
                "temperature",
                1.0,
            ),

        # ----------------------------------------------------------
        # Confidence gate
        # ----------------------------------------------------------

        gate_hidden_dim=
            gate_cfg.get(
                "gate_hidden_dim",
                128,
            ),

        gate_dropout=
            gate_cfg.get(
                "dropout",
                0.0,
            ),
    )

    return model


# ======================================================================
# Checkpoint discovery
# ======================================================================


def get_checkpoint_path(
    checkpoint_dir: Path,
    fold: int,
) -> Path:
    """
    Locate a fold-specific best checkpoint.

    Preferred structure:

        checkpoints/fold_1/hvnet_fold1_best.pth

    A root-level fallback is also supported:

        checkpoints/hvnet_fold1_best.pth
    """

    preferred_path = (
        checkpoint_dir
        / f"fold_{fold}"
        / f"hvnet_fold{fold}_best.pth"
    )

    if preferred_path.exists():
        return preferred_path

    fallback_path = (
        checkpoint_dir
        / f"hvnet_fold{fold}_best.pth"
    )

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Checkpoint for fold {fold} was not found.\n"
        f"Checked:\n"
        f"  {preferred_path}\n"
        f"  {fallback_path}"
    )


# ======================================================================
# Checkpoint loading
# ======================================================================


def load_fold_model(
    checkpoint_path: Path,
    fallback_config: Dict,
    device: torch.device,
) -> Tuple[
    HVNet,
    Dict,
]:
    """
    Load one fold-specific model.

    If the checkpoint contains the original training configuration,
    that configuration is preferred when reconstructing the model.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    checkpoint_config = checkpoint.get(
        "config",
        fallback_config,
    )

    model = build_model_from_config(
        checkpoint_config
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            f"'model_state_dict' is missing from {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ======================================================================
# Evaluation dataset
# ======================================================================


def build_dataset(
    metadata_file: str,
    root_dir: Optional[str],
    config: Dict,
) -> HVNetDataset:
    """
    Build an evaluation-only HVNet dataset.
    """

    data_cfg = config.get(
        "data",
        {},
    )

    preprocessing_cfg = config.get(
        "preprocessing",
        {},
    )

    image_size = data_cfg.get(
        "image_size",
        preprocessing_cfg.get(
            "resize",
            {},
        ).get(
            "height",
            224,
        ),
    )

    transform = build_eval_transform(
        image_size=image_size,
    )

    dataset = HVNetDataset(
        metadata_file=
            metadata_file,

        root_dir=
            root_dir,

        transform=
            transform,

        image_column=
            data_cfg.get(
                "image_column",
                "image_path",
            ),

        patient_id_column=
            data_cfg.get(
                "patient_id_column",
                "patient_id",
            ),

        iop_column=
            "iop",

        acd_column=
            "acd",

        cag_column=
            "cag",

        label_column=
            data_cfg.get(
                "label_column",
                "label",
            ),
    )

    return dataset


# ======================================================================
# Fixed internal-test selection
# ======================================================================


def get_internal_test_indices(
    dataset: HVNetDataset,
    split_column: str,
    internal_test_value: str,
) -> np.ndarray:
    """
    Identify the fixed internal-test subset.

    Unlike train.py, evaluation does NOT generate a new test split.
    """

    metadata = (
        dataset.get_metadata()
    )

    if (
        split_column
        not in metadata.columns
    ):
        raise ValueError(
            f"Internal metadata does not contain the required "
            f"split column '{split_column}'.\n\n"
            "Evaluation must use the original fixed internal-test "
            "assignment. Do not generate a new test split at this stage."
        )

    # --------------------------------------------------------------
    # Check that each patient has only one split assignment.
    # --------------------------------------------------------------

    patient_id_column = (
        dataset.patient_id_column
    )

    patient_split_counts = (
        metadata.groupby(
            patient_id_column
        )[
            split_column
        ]
        .nunique()
    )

    inconsistent_patients = (
        patient_split_counts[
            patient_split_counts
            > 1
        ]
    )

    if len(
        inconsistent_patients
    ) > 0:

        raise RuntimeError(
            "At least one patient appears in more than one "
            "internal cohort split. Patient-level leakage detected."
        )

    split_values = (
        metadata[
            split_column
        ]
        .astype(str)
    )

    indices = np.where(
        split_values
        == internal_test_value
    )[0]

    if len(
        indices
    ) == 0:

        raise ValueError(
            f"No samples with "
            f"{split_column}='{internal_test_value}' "
            "were found."
        )

    return np.asarray(
        indices
    )


# ======================================================================
# DataLoader
# ======================================================================


def build_eval_loader(
    dataset: HVNetDataset,
    indices: Optional[Sequence[int]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """
    Construct deterministic evaluation DataLoader.
    """

    if indices is None:

        evaluation_dataset = (
            dataset
        )

    else:

        evaluation_dataset = Subset(
            dataset,
            indices,
        )

    return DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=False,
    )


# ======================================================================
# Evaluate one checkpoint on one cohort
# ======================================================================


@torch.no_grad()
def evaluate_model(
    model: HVNet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
) -> Tuple[
    Dict,
    pd.DataFrame,
]:
    """
    Evaluate one fold-specific model on one complete cohort.

    Metrics are calculated after accumulating all predictions from the
    complete cohort.

    This is important for AUC, specificity, and Cohen's kappa.
    """

    model.eval()

    tracker = HVNetMetricTracker(
        num_classes=num_classes,
        scale=100.0,
    )

    prediction_records = []

    for batch in loader:

        # ----------------------------------------------------------
        # Inputs
        # ----------------------------------------------------------

        image = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        iop = batch[
            "iop"
        ].to(
            device,
            non_blocking=True,
        )

        acd = batch[
            "acd"
        ].to(
            device,
            non_blocking=True,
        )

        cag = batch[
            "cag"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------------
        # HVNet forward
        # ----------------------------------------------------------

        outputs = model(
            image=image,
            iop=iop,
            acd=acd,
            cag=cag,
            return_auxiliary=False,
        )

        probabilities = outputs[
            "p_final"
        ]

        # ----------------------------------------------------------
        # Cohort-level metric accumulation
        # ----------------------------------------------------------

        tracker.update(
            targets=targets,
            probabilities=probabilities,
        )

        # ----------------------------------------------------------
        # Save sample-level predictions
        # ----------------------------------------------------------

        probabilities_np = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

        predictions_np = np.argmax(
            probabilities_np,
            axis=1,
        )

        targets_np = (
            targets
            .detach()
            .cpu()
            .numpy()
        )

        batch_size_current = (
            targets_np.shape[0]
        )

        for sample_index in range(
            batch_size_current
        ):

            record = {
                "patient_id":
                    str(
                        batch[
                            "patient_id"
                        ][
                            sample_index
                        ]
                    ),

                "image_path":
                    str(
                        batch[
                            "image_path"
                        ][
                            sample_index
                        ]
                    ),

                "label":
                    int(
                        targets_np[
                            sample_index
                        ]
                    ),

                "prediction":
                    int(
                        predictions_np[
                            sample_index
                        ]
                    ),
            }

            # ------------------------------------------------------
            # Save class probabilities
            # ------------------------------------------------------

            for class_index in range(
                num_classes
            ):

                if (
                    class_index
                    < len(
                        class_names
                    )
                ):

                    class_name = (
                        class_names[
                            class_index
                        ]
                    )

                else:

                    class_name = (
                        f"class_{class_index}"
                    )

                probability_key = (
                    "prob_"
                    + str(
                        class_name
                    ).replace(
                        " ",
                        "_",
                    )
                )

                record[
                    probability_key
                ] = float(
                    probabilities_np[
                        sample_index,
                        class_index,
                    ]
                )

            # ------------------------------------------------------
            # Gate weights
            # ------------------------------------------------------

            if (
                "w_clin"
                in outputs
            ):

                record[
                    "w_clin"
                ] = float(
                    outputs[
                        "w_clin"
                    ][
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

            if (
                "w_vis"
                in outputs
            ):

                record[
                    "w_vis"
                ] = float(
                    outputs[
                        "w_vis"
                    ][
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

            if (
                "w_ds"
                in outputs
            ):

                record[
                    "w_ds"
                ] = float(
                    outputs[
                        "w_ds"
                    ][
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

            if (
                "conflict"
                in outputs
            ):

                record[
                    "ds_conflict"
                ] = float(
                    outputs[
                        "conflict"
                    ][
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

            prediction_records.append(
                record
            )

    if len(
        tracker
    ) == 0:

        raise RuntimeError(
            "Evaluation DataLoader contains no samples."
        )

    metrics = tracker.compute(
        return_per_class=True,
        return_confusion_matrix=True,
    )

    predictions_df = pd.DataFrame(
        prediction_records
    )

    return (
        metrics,
        predictions_df,
    )


# ======================================================================
# Scalar metric extraction
# ======================================================================


def extract_scalar_metrics(
    metrics: Dict,
) -> Dict[str, float]:
    """
    Retain only the four manuscript-level scalar metrics.
    """

    names = [
        "accuracy",
        "specificity",
        "auc",
        "kappa",
    ]

    return {
        name:
            float(
                metrics[
                    name
                ]
            )

        for name in names
    }


# ======================================================================
# Save fold metrics
# ======================================================================


def save_fold_metrics(
    metrics: Dict,
    output_path: Path,
) -> None:
    """
    Save one fold's detailed evaluation metrics.
    """

    serializable = {}

    for key, value in metrics.items():

        if isinstance(
            value,
            np.ndarray,
        ):

            serializable[
                key
            ] = value.tolist()

        elif np.isscalar(
            value
        ):

            serializable[
                key
            ] = float(
                value
            )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable,
            file,
            indent=2,
        )


# ======================================================================
# Cohort summary
# ======================================================================


def summarize_cohort(
    fold_metrics: List[
        Dict[str, float]
    ],
    cohort_name: str,
    output_dir: Path,
) -> Dict:
    """
    Compute and save mean +/- SD across fold-specific models.
    """

    summary = summarize_fold_metrics(
        fold_metrics=fold_metrics,
        metric_names=[
            "accuracy",
            "specificity",
            "auc",
            "kappa",
        ],
    )

    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    json_path = (
        output_dir
        / f"{cohort_name}_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    rows = []

    for metric_name, stats in summary.items():

        rows.append(
            {
                "metric":
                    metric_name,

                "mean":
                    stats[
                        "mean"
                    ],

                "std":
                    stats[
                        "std"
                    ],

                "formatted":
                    format_mean_std(
                        stats[
                            "mean"
                        ],
                        stats[
                            "std"
                        ],
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        / f"{cohort_name}_summary.csv",
        index=False,
    )

    return summary


# ======================================================================
# Console summary
# ======================================================================


def print_summary(
    title: str,
    summary: Dict,
) -> None:
    """
    Print manuscript-style mean +/- SD results.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )

    display_names = {
        "accuracy":
            "Accuracy",

        "specificity":
            "Macro-Specificity",

        "auc":
            "Macro OvR AUC",

        "kappa":
            "Cohen's Kappa",
    }

    for key in [
        "accuracy",
        "specificity",
        "auc",
        "kappa",
    ]:

        statistics = summary[
            key
        ]

        formatted = format_mean_std(
            statistics[
                "mean"
            ],
            statistics[
                "std"
            ],
            decimals=2,
        )

        print(
            f"{display_names[key]:20s}: "
            f"{formatted}"
        )


# ======================================================================
# Command-line arguments
# ======================================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate five fold-specific HVNet checkpoints "
            "on fixed test cohorts."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to default YAML configuration.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing fold-specific checkpoints.",
    )

    # ------------------------------------------------------------------
    # Internal test cohort
    # ------------------------------------------------------------------

    parser.add_argument(
        "--internal-metadata",
        type=str,
        default=None,
        help=(
            "Metadata CSV containing the internal cohort "
            "and its fixed split assignment. "
            "If omitted, data.metadata_file from YAML is used."
        ),
    )

    parser.add_argument(
        "--internal-root-dir",
        type=str,
        default=None,
        help="Root directory containing internal CFP images.",
    )

    parser.add_argument(
        "--split-column",
        type=str,
        default="split",
        help=(
            "Metadata column defining the fixed internal "
            "development/test assignment."
        ),
    )

    parser.add_argument(
        "--internal-test-value",
        type=str,
        default="internal_test",
        help=(
            "Value identifying the fixed internal test samples."
        ),
    )

    # ------------------------------------------------------------------
    # Held-out cohort
    # ------------------------------------------------------------------

    parser.add_argument(
        "--heldout-metadata",
        type=str,
        default=None,
        help=(
            "Metadata CSV for the held-out community-referral cohort. "
            "If omitted, held-out evaluation is skipped."
        ),
    )

    parser.add_argument(
        "--heldout-root-dir",
        type=str,
        default=None,
        help=(
            "Root directory containing held-out community-referral "
            "CFP images."
        ),
    )

    # ------------------------------------------------------------------
    # Evaluation settings
    # ------------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            3,
            4,
            5,
        ],
        help="Fold-specific checkpoints to evaluate.",
    )

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================


def main():

    args = parse_args()

    # ==============================================================
    # Configuration
    # ==============================================================

    config = load_config(
        args.config
    )

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    data_cfg = config.get(
        "data",
        {},
    )

    seed = experiment_cfg.get(
        "seed",
        42,
    )

    num_classes = experiment_cfg.get(
        "num_classes",
        3,
    )

    class_names = experiment_cfg.get(
        "class_names",
        [
            "Normal",
            "OAG",
            "ACG",
        ],
    )

    set_random_seed(
        seed
    )

    # ==============================================================
    # Device
    # ==============================================================

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "Evaluation device:",
        device,
    )

    # ==============================================================
    # Output directory
    # ==============================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    # ==============================================================
    # Internal test dataset
    # ==============================================================

    internal_metadata = (
        args.internal_metadata
        if args.internal_metadata
        is not None
        else data_cfg.get(
            "metadata_file"
        )
    )

    internal_dataset = None
    internal_loader = None
    internal_indices = None

    if internal_metadata is not None:

        internal_root_dir = (
            args.internal_root_dir
            if args.internal_root_dir
            is not None
            else data_cfg.get(
                "root_dir"
            )
        )

        internal_dataset = build_dataset(
            metadata_file=
                internal_metadata,

            root_dir=
                internal_root_dir,

            config=
                config,
        )

        internal_indices = (
            get_internal_test_indices(
                dataset=
                    internal_dataset,

                split_column=
                    args.split_column,

                internal_test_value=
                    args.internal_test_value,
            )
        )

        internal_loader = (
            build_eval_loader(
                dataset=
                    internal_dataset,

                indices=
                    internal_indices,

                batch_size=
                    args.batch_size,

                num_workers=
                    args.num_workers,

                device=
                    device,
            )
        )

        print(
            "\nFixed internal test cohort:"
        )

        print(
            f"Eyes: {len(internal_indices)}"
        )

        internal_patient_ids = {
            str(x)
            for x in (
                internal_dataset
                .get_metadata()
                .iloc[
                    internal_indices
                ][
                    internal_dataset
                    .patient_id_column
                ]
                .tolist()
            )
        }

        print(
            f"Patients: "
            f"{len(internal_patient_ids)}"
        )

    else:

        print(
            "\nNo internal metadata provided. "
            "Internal-test evaluation will be skipped."
        )

    # ==============================================================
    # Held-out community-referral dataset
    # ==============================================================

    heldout_dataset = None
    heldout_loader = None

    if (
        args.heldout_metadata
        is not None
    ):

        heldout_dataset = build_dataset(
            metadata_file=
                args.heldout_metadata,

            root_dir=
                args.heldout_root_dir,

            config=
                config,
        )

        heldout_loader = build_eval_loader(
            dataset=
                heldout_dataset,

            indices=
                None,

            batch_size=
                args.batch_size,

            num_workers=
                args.num_workers,

            device=
                device,
        )

        print(
            "\nHeld-out community-referral cohort:"
        )

        print(
            f"Eyes: {len(heldout_dataset)}"
        )

        heldout_patient_ids = set(
            heldout_dataset
            .get_patient_ids()
            .astype(str)
            .tolist()
        )

        print(
            f"Patients: "
            f"{len(heldout_patient_ids)}"
        )

    else:

        print(
            "\nNo held-out metadata provided. "
            "Held-out evaluation will be skipped."
        )

    # --------------------------------------------------------------
    # At least one cohort is required.
    # --------------------------------------------------------------

    if (
        internal_loader is None
        and heldout_loader is None
    ):

        raise ValueError(
            "No evaluation cohort was provided."
        )

    # ==============================================================
    # Fold-specific evaluation
    # ==============================================================

    internal_fold_metrics = []
    heldout_fold_metrics = []

    all_results = {
        "folds": {},
    }

    for fold in args.folds:

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"Evaluating Fold {fold}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------------

        checkpoint_path = (
            get_checkpoint_path(
                checkpoint_dir=
                    checkpoint_dir,

                fold=
                    fold,
            )
        )

        print(
            "Checkpoint:"
        )

        print(
            checkpoint_path
        )

        (
            model,
            checkpoint,
        ) = load_fold_model(
            checkpoint_path=
                checkpoint_path,

            fallback_config=
                config,

            device=
                device,
        )

        checkpoint_fold = (
            checkpoint.get(
                "fold"
            )
        )

        if (
            checkpoint_fold
            is not None
            and int(
                checkpoint_fold
            )
            != fold
        ):

            raise RuntimeError(
                f"Checkpoint fold mismatch: "
                f"requested fold {fold}, "
                f"checkpoint reports fold {checkpoint_fold}."
            )

        fold_result = {
            "checkpoint":
                str(
                    checkpoint_path
                ),

            "checkpoint_epoch":
                checkpoint.get(
                    "epoch"
                ),
        }

        # ==========================================================
        # Internal test
        # ==========================================================

        if (
            internal_loader
            is not None
        ):

            (
                internal_metrics,
                internal_predictions,
            ) = evaluate_model(
                model=model,
                loader=
                    internal_loader,
                device=device,
                num_classes=
                    num_classes,
                class_names=
                    class_names,
            )

            scalar_internal = (
                extract_scalar_metrics(
                    internal_metrics
                )
            )

            internal_fold_metrics.append(
                scalar_internal
            )

            fold_internal_dir = (
                output_dir
                / "internal_test"
                / f"fold_{fold}"
            )

            fold_internal_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            internal_predictions.to_csv(
                fold_internal_dir
                / "predictions.csv",
                index=False,
            )

            save_fold_metrics(
                metrics=
                    internal_metrics,

                output_path=
                    fold_internal_dir
                    / "metrics.json",
            )

            fold_result[
                "internal_test"
            ] = scalar_internal

            print(
                "\nInternal Test:"
            )

            print(
                f"  Accuracy:          "
                f"{scalar_internal['accuracy']:.2f}"
            )

            print(
                f"  Macro-Specificity: "
                f"{scalar_internal['specificity']:.2f}"
            )

            print(
                f"  Macro OvR AUC:     "
                f"{scalar_internal['auc']:.2f}"
            )

            print(
                f"  Cohen's Kappa:     "
                f"{scalar_internal['kappa']:.2f}"
            )

        # ==========================================================
        # Held-out community-referral cohort
        # ==========================================================

        if (
            heldout_loader
            is not None
        ):

            (
                heldout_metrics,
                heldout_predictions,
            ) = evaluate_model(
                model=model,
                loader=
                    heldout_loader,
                device=device,
                num_classes=
                    num_classes,
                class_names=
                    class_names,
            )

            scalar_heldout = (
                extract_scalar_metrics(
                    heldout_metrics
                )
            )

            heldout_fold_metrics.append(
                scalar_heldout
            )

            fold_heldout_dir = (
                output_dir
                / "heldout_referral"
                / f"fold_{fold}"
            )

            fold_heldout_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            heldout_predictions.to_csv(
                fold_heldout_dir
                / "predictions.csv",
                index=False,
            )

            save_fold_metrics(
                metrics=
                    heldout_metrics,

                output_path=
                    fold_heldout_dir
                    / "metrics.json",
            )

            fold_result[
                "heldout_referral"
            ] = scalar_heldout

            print(
                "\nHeld-out Community-Referral:"
            )

            print(
                f"  Accuracy:          "
                f"{scalar_heldout['accuracy']:.2f}"
            )

            print(
                f"  Macro-Specificity: "
                f"{scalar_heldout['specificity']:.2f}"
            )

            print(
                f"  Macro OvR AUC:     "
                f"{scalar_heldout['auc']:.2f}"
            )

            print(
                f"  Cohen's Kappa:     "
                f"{scalar_heldout['kappa']:.2f}"
            )

        all_results[
            "folds"
        ][
            str(
                fold
            )
        ] = fold_result

        # ----------------------------------------------------------
        # Release model before loading the next fold.
        # ----------------------------------------------------------

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ==============================================================
    # Mean +/- SD across fold-specific models
    # ==============================================================

    if len(
        internal_fold_metrics
    ) > 0:

        internal_summary = (
            summarize_cohort(
                fold_metrics=
                    internal_fold_metrics,

                cohort_name=
                    "internal_test",

                output_dir=
                    output_dir,
            )
        )

        all_results[
            "internal_test_summary"
        ] = internal_summary

        print_summary(
            title=(
                "Fixed Internal Test Cohort "
                "(mean ± SD across fold-specific models)"
            ),
            summary=
                internal_summary,
        )

    if len(
        heldout_fold_metrics
    ) > 0:

        heldout_summary = (
            summarize_cohort(
                fold_metrics=
                    heldout_fold_metrics,

                cohort_name=
                    "heldout_referral",

                output_dir=
                    output_dir,
            )
        )

        all_results[
            "heldout_referral_summary"
        ] = heldout_summary

        print_summary(
            title=(
                "Held-out Community-Referral Cohort "
                "(mean ± SD across fold-specific models)"
            ),
            summary=
                heldout_summary,
        )

    # ==============================================================
    # Save combined summary
    # ==============================================================

    all_results[
        "evaluation_protocol"
    ] = {
        "num_fold_specific_models":
            len(
                args.folds
            ),

        "folds":
            args.folds,

        "metrics_scale":
            "0-100",

        "aggregation":
            "mean_and_standard_deviation_across_fold_specific_models",

        "model_ensemble":
            False,

        "internal_test_used_for_checkpoint_selection":
            False,

        "heldout_used_for_checkpoint_selection":
            False,
    }

    with open(
        output_dir
        / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HVNet evaluation complete."
    )

    print(
        "=" * 70
    )

    print(
        f"Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
