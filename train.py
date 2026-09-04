"""
Training script for HVNet.

HVNet:
    Hypothesis-and-Verify Network for three-class glaucoma triage.

This script implements the training protocol described in the manuscript.

=======================================================================
Training protocol
=======================================================================

Internal cohort:

    2,281 eyes / 1,402 participants

Patient-level partition:

    80% development subset
    20% fixed internal test subset

The fixed internal test subset is NEVER used for:

    - model training
    - validation
    - hyperparameter selection
    - checkpoint selection

Within the development subset:

    5-fold patient-level cross-validation

For each fold:

    4 folds -> training
    1 fold  -> validation

The best checkpoint is selected according to validation accuracy.

Each selected fold-specific model can subsequently be evaluated on:

    1. the fixed internal test subset
    2. the held-out community-referral cohort

using evaluate.py.

=======================================================================
Optimization
=======================================================================

Epochs:
    80

Batch size:
    32

Optimizer:
    AdamW

Weight decay:
    0.01

Learning rates:
    ConvNeXt backbone: 5e-5
    Remaining modules: 5e-4

Scheduler:
    Cosine annealing
    eta_min = 5e-7

Backbone freezing:
    Epochs 1--5: frozen
    Epochs 6--80: trainable

Gradient clipping:
    max norm = 5.0

Random seed:
    42

=======================================================================
Important
=======================================================================

For exact reproduction of the manuscript split, the original metadata
should contain the fixed development/internal-test assignment.

If no preassigned split column is available, this public implementation
can generate a deterministic patient-level 80/20 split as a fallback.

The fallback split is intended for code usability and should NOT be
assumed to reproduce the exact manuscript cohort partition.
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

from sklearn.model_selection import StratifiedGroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from datasets.dataset_template import HVNetDataset
from models.hvnet import HVNet
from utils.losses import HVNetLoss
from utils.metrics import HVNetMetricTracker
from utils.transforms import (
    build_eval_transform,
    build_train_transform,
)


# ======================================================================
# Configuration utilities
# ======================================================================


def load_config(
    config_path: str,
) -> Dict:
    """
    Load YAML configuration.
    """

    config_path = Path(config_path)

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
    Set random seeds used by Python, NumPy, and PyTorch.

    Exact bitwise determinism can still depend on CUDA kernels,
    GPU architecture, and PyTorch/CUDA versions.
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


def seed_worker(
    worker_id: int,
) -> None:
    """
    Initialize DataLoader worker random seeds.
    """

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


# ======================================================================
# Dataset split utilities
# ======================================================================


def create_fallback_internal_split(
    metadata: pd.DataFrame,
    patient_id_column: str,
    label_column: str,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate a deterministic patient-level approximately 80/20 split.

    This function is used ONLY when the metadata file does not already
    contain the manuscript's fixed development/internal-test assignment.

    A five-way StratifiedGroupKFold is used and one fold is reserved
    as the fixed internal test set.

    This provides approximately:

        80% development
        20% internal test

    while ensuring that samples from the same patient remain together.

    Returns
    -------
    development_indices
    internal_test_indices
    """

    groups = (
        metadata[
            patient_id_column
        ]
        .astype(str)
        .to_numpy()
    )

    labels = (
        metadata[
            label_column
        ]
        .astype(int)
        .to_numpy()
    )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    all_indices = np.arange(
        len(metadata)
    )

    # The first deterministic fold is used as the fallback
    # fixed internal test partition.
    development_indices, internal_test_indices = next(
        splitter.split(
            X=all_indices,
            y=labels,
            groups=groups,
        )
    )

    return (
        np.asarray(
            development_indices
        ),
        np.asarray(
            internal_test_indices
        ),
    )


def resolve_internal_split(
    metadata: pd.DataFrame,
    patient_id_column: str,
    label_column: str,
    split_column: str,
    development_value: str,
    internal_test_value: str,
    seed: int,
    output_dir: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Obtain development and fixed internal-test indices.

    Preferred workflow
    ------------------
    If metadata contains `split_column`, use the preassigned manuscript
    split directly.

    Fallback workflow
    -----------------
    If the column is absent, generate a deterministic patient-level
    approximately 80/20 split and save the generated assignment.
    """

    if split_column in metadata.columns:

        split_values = (
            metadata[
                split_column
            ]
            .astype(str)
        )

        development_indices = np.where(
            split_values
            == development_value
        )[0]

        internal_test_indices = np.where(
            split_values
            == internal_test_value
        )[0]

        if len(
            development_indices
        ) == 0:

            raise ValueError(
                f"No samples with "
                f"{split_column}='{development_value}' "
                "were found."
            )

        if len(
            internal_test_indices
        ) == 0:

            raise ValueError(
                f"No samples with "
                f"{split_column}='{internal_test_value}' "
                "were found."
            )

        print(
            "\nUsing preassigned internal cohort split "
            "from metadata."
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            f"Metadata does not contain '{split_column}'."
        )

        print(
            "Generating a deterministic patient-level "
            "80/20 split as a fallback."
        )

        print(
            "This fallback should NOT be assumed to reproduce "
            "the exact manuscript partition.\n"
        )

        (
            development_indices,
            internal_test_indices,
        ) = create_fallback_internal_split(
            metadata=metadata,
            patient_id_column=patient_id_column,
            label_column=label_column,
            seed=seed,
        )

        generated_metadata = (
            metadata.copy()
        )

        generated_metadata[
            "generated_split"
        ] = "development"

        generated_metadata.loc[
            internal_test_indices,
            "generated_split",
        ] = "internal_test"

        generated_path = (
            output_dir
            / "generated_internal_split.csv"
        )

        generated_metadata.to_csv(
            generated_path,
            index=False,
        )

        print(
            "Generated split saved to:"
        )

        print(
            generated_path
        )

    # --------------------------------------------------------------
    # Patient leakage check
    # --------------------------------------------------------------

    development_patients = set(
        metadata.iloc[
            development_indices
        ][
            patient_id_column
        ]
        .astype(str)
        .tolist()
    )

    internal_test_patients = set(
        metadata.iloc[
            internal_test_indices
        ][
            patient_id_column
        ]
        .astype(str)
        .tolist()
    )

    overlap = (
        development_patients
        & internal_test_patients
    )

    if overlap:
        raise RuntimeError(
            "Patient leakage detected between development "
            "and internal-test subsets."
        )

    return (
        development_indices,
        internal_test_indices,
    )


# ======================================================================
# Five-fold development split
# ======================================================================


def build_development_folds(
    metadata: pd.DataFrame,
    development_indices: np.ndarray,
    patient_id_column: str,
    label_column: str,
    num_folds: int,
    seed: int,
) -> List[
    Tuple[np.ndarray, np.ndarray]
]:
    """
    Generate patient-level folds within the development subset.

    Implements:

        5-fold CV within development cohort

    with all eyes from the same patient assigned to the same fold.
    """

    development_metadata = (
        metadata.iloc[
            development_indices
        ]
        .reset_index(
            drop=False
        )
    )

    labels = (
        development_metadata[
            label_column
        ]
        .astype(int)
        .to_numpy()
    )

    groups = (
        development_metadata[
            patient_id_column
        ]
        .astype(str)
        .to_numpy()
    )

    local_indices = np.arange(
        len(
            development_metadata
        )
    )

    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )

    folds = []

    for (
        local_train_indices,
        local_val_indices,
    ) in splitter.split(
        X=local_indices,
        y=labels,
        groups=groups,
    ):

        # Convert development-local indices back to
        # original metadata indices.
        train_indices = (
            development_metadata.iloc[
                local_train_indices
            ][
                "index"
            ]
            .to_numpy()
        )

        val_indices = (
            development_metadata.iloc[
                local_val_indices
            ][
                "index"
            ]
            .to_numpy()
        )

        # ----------------------------------------------------------
        # Patient leakage check
        # ----------------------------------------------------------

        train_patients = set(
            metadata.iloc[
                train_indices
            ][
                patient_id_column
            ]
            .astype(str)
            .tolist()
        )

        val_patients = set(
            metadata.iloc[
                val_indices
            ][
                patient_id_column
            ]
            .astype(str)
            .tolist()
        )

        if (
            train_patients
            & val_patients
        ):

            raise RuntimeError(
                "Patient leakage detected between training "
                "and validation folds."
            )

        folds.append(
            (
                np.asarray(
                    train_indices
                ),
                np.asarray(
                    val_indices
                ),
            )
        )

    return folds


# ======================================================================
# Model construction
# ======================================================================


def build_model_from_config(
    config: Dict,
) -> HVNet:
    """
    Construct HVNet using available configuration values.

    Parameters not explicitly specified in the YAML fall back to
    models/hvnet.py defaults.

    Implementation-level defaults must ultimately be checked against
    the original experimental code before final public release.
    """

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

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    out_indices = backbone_cfg.get(
        "out_indices",
        [1, 2],
    )

    model = HVNet(
        num_classes=experiment_cfg.get(
            "num_classes",
            3,
        ),

        backbone_name=backbone_cfg.get(
            "name",
            "convnext_tiny",
        ),

        pretrained=backbone_cfg.get(
            "pretrained",
            True,
        ),

        backbone_out_indices=tuple(
            out_indices
        ),

        clinical_feature_dim=
            clinical_cfg.get(
                "clinical_feature_dim",
                256,
            ),

        visual_feature_dim=
            visual_cfg.get(
                "visual_feature_dim",
                256,
            ),

        num_local_patches=
            visual_cfg.get(
                "local_branch",
                {},
            ).get(
                "num_local_patches",
                4,
            ),

        ds_temperature=
            ds_cfg.get(
                "temperature",
                1.0,
            ),

        gate_hidden_dim=
            gate_cfg.get(
                "gate_hidden_dim",
                128,
            ),
    )

    return model


# ======================================================================
# Optimizer
# ======================================================================


def build_optimizer(
    model: HVNet,
    config: Dict,
) -> AdamW:
    """
    Build AdamW with separate learning rates for:

        1. ConvNeXt backbone
        2. Remaining HVNet modules

    Manuscript:

        backbone LR = 5e-5
        other LR    = 5e-4
        weight decay = 0.01
    """

    optimizer_cfg = config.get(
        "optimizer",
        {},
    )

    lr_cfg = optimizer_cfg.get(
        "learning_rates",
        {},
    )

    backbone_lr = lr_cfg.get(
        "backbone",
        5.0e-5,
    )

    other_lr = lr_cfg.get(
        "remaining_modules",
        5.0e-4,
    )

    weight_decay = optimizer_cfg.get(
        "weight_decay",
        0.01,
    )

    beta1 = optimizer_cfg.get(
        "beta1",
        0.9,
    )

    beta2 = optimizer_cfg.get(
        "beta2",
        0.999,
    )

    backbone_parameters = list(
        model.visual_backbone.parameters()
    )

    backbone_parameter_ids = {
        id(parameter)
        for parameter in backbone_parameters
    }

    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter)
        not in backbone_parameter_ids
    ]

    optimizer = AdamW(
        [
            {
                "params":
                    backbone_parameters,

                "lr":
                    backbone_lr,

                "name":
                    "backbone",
            },
            {
                "params":
                    other_parameters,

                "lr":
                    other_lr,

                "name":
                    "remaining_modules",
            },
        ],
        betas=(
            beta1,
            beta2,
        ),
        weight_decay=weight_decay,
    )

    return optimizer


# ======================================================================
# DataLoader
# ======================================================================


def build_loader(
    dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Construct DataLoader for a subset of the dataset.
    """

    subset = Subset(
        dataset,
        indices,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    return loader


# ======================================================================
# One training epoch
# ======================================================================


def train_one_epoch(
    model: HVNet,
    loader: DataLoader,
    criterion: HVNetLoss,
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_max_norm: float,
) -> Dict[str, float]:
    """
    Train HVNet for one epoch.
    """

    model.train()

    total_samples = 0

    accumulated = {
        "total_loss": 0.0,
        "loss_final": 0.0,
        "loss_clin": 0.0,
        "loss_vis": 0.0,
        "kl_clin": 0.0,
        "kl_vis": 0.0,
    }

    for batch in loader:

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

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            image=image,
            iop=iop,
            acd=acd,
            cag=cag,
            return_auxiliary=False,
        )

        losses = criterion(
            outputs=outputs,
            targets=targets,
        )

        total_loss = losses[
            "total_loss"
        ]

        total_loss.backward()

        # ----------------------------------------------------------
        # Gradient clipping
        #
        # Manuscript:
        # max norm = 5
        # ----------------------------------------------------------

        clip_grad_norm_(
            model.parameters(),
            max_norm=
                gradient_clip_max_norm,
        )

        optimizer.step()

        batch_size = (
            targets.size(0)
        )

        total_samples += batch_size

        for key in accumulated:

            accumulated[
                key
            ] += (
                float(
                    losses[
                        key
                    ].detach().item()
                )
                * batch_size
            )

    if total_samples == 0:
        raise RuntimeError(
            "Training DataLoader contains no samples."
        )

    return {
        key:
            value
            / total_samples

        for key, value
        in accumulated.items()
    }


# ======================================================================
# Validation
# ======================================================================


@torch.no_grad()
def validate(
    model: HVNet,
    loader: DataLoader,
    criterion: HVNetLoss,
    device: torch.device,
    num_classes: int,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
]:
    """
    Evaluate one complete validation cohort.

    Predictions are accumulated over the full validation fold before
    Accuracy, Specificity, AUC, and Kappa are calculated.
    """

    model.eval()

    tracker = HVNetMetricTracker(
        num_classes=num_classes,
        scale=100.0,
    )

    total_samples = 0

    accumulated = {
        "total_loss": 0.0,
        "loss_final": 0.0,
        "loss_clin": 0.0,
        "loss_vis": 0.0,
        "kl_clin": 0.0,
        "kl_vis": 0.0,
    }

    for batch in loader:

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

        outputs = model(
            image=image,
            iop=iop,
            acd=acd,
            cag=cag,
            return_auxiliary=False,
        )

        losses = criterion(
            outputs=outputs,
            targets=targets,
        )

        tracker.update(
            targets=targets,
            probabilities=
                outputs[
                    "p_final"
                ],
        )

        batch_size = (
            targets.size(0)
        )

        total_samples += batch_size

        for key in accumulated:

            accumulated[
                key
            ] += (
                float(
                    losses[
                        key
                    ].detach().item()
                )
                * batch_size
            )

    if total_samples == 0:
        raise RuntimeError(
            "Validation DataLoader contains no samples."
        )

    validation_losses = {
        key:
            value
            / total_samples

        for key, value
        in accumulated.items()
    }

    validation_metrics = (
        tracker.compute()
    )

    return (
        validation_losses,
        validation_metrics,
    )


# ======================================================================
# Checkpoint
# ======================================================================


def save_checkpoint(
    path: Path,
    model: HVNet,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    fold: int,
    validation_metrics: Dict,
    config: Dict,
) -> None:
    """
    Save one fold-specific best checkpoint.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "fold":
            fold,

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "validation_metrics":
            validation_metrics,

        "config":
            config,
    }

    torch.save(
        checkpoint,
        path,
    )


# ======================================================================
# Train one CV fold
# ======================================================================


def train_fold(
    fold_number: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_dataset: HVNetDataset,
    val_dataset: HVNetDataset,
    config: Dict,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> Dict:
    """
    Train one fold-specific HVNet model.
    """

    training_cfg = config.get(
        "training",
        {},
    )

    scheduler_cfg = config.get(
        "scheduler",
        {},
    )

    loss_cfg = config.get(
        "loss",
        {},
    )

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    epochs = training_cfg.get(
        "epochs",
        80,
    )

    batch_size = training_cfg.get(
        "batch_size",
        32,
    )

    freeze_epochs = (
        training_cfg
        .get(
            "freeze_backbone",
            {},
        )
        .get(
            "epochs",
            5,
        )
    )

    gradient_clip = (
        training_cfg
        .get(
            "gradient_clipping",
            {},
        )
        .get(
            "max_norm",
            5.0,
        )
    )

    num_classes = experiment_cfg.get(
        "num_classes",
        3,
    )

    lambda_aux = (
        loss_cfg
        .get(
            "auxiliary",
            {},
        )
        .get(
            "weight",
            0.3,
        )
    )

    lambda_edl = (
        loss_cfg
        .get(
            "evidential",
            {},
        )
        .get(
            "weight",
            0.1,
        )
    )

    eta_min = scheduler_cfg.get(
        "eta_min",
        5.0e-7,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Fold {fold_number}"
    )

    print(
        "=" * 70
    )

    print(
        f"Training eyes:   {len(train_indices)}"
    )

    print(
        f"Validation eyes: {len(val_indices)}"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = build_model_from_config(
        config
    )

    model.to(
        device
    )

    # --------------------------------------------------------------
    # Optimizer must contain backbone parameters BEFORE the backbone
    # is frozen, so that the same optimizer can continue to update
    # them after epoch 5.
    # --------------------------------------------------------------

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    # --------------------------------------------------------------
    # Freeze backbone before epoch 1.
    # --------------------------------------------------------------

    if freeze_epochs > 0:

        model.freeze_backbone()

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    criterion = HVNetLoss(
        lambda_aux=lambda_aux,
        lambda_edl=lambda_edl,
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    pin_memory = (
        device.type
        == "cuda"
    )

    train_loader = build_loader(
        dataset=train_dataset,
        indices=train_indices,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=
            seed
            + fold_number,
        pin_memory=pin_memory,
    )

    val_loader = build_loader(
        dataset=val_dataset,
        indices=val_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=
            seed
            + fold_number,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------------
    # Fold outputs
    # --------------------------------------------------------------

    fold_directory = (
        output_dir
        / f"fold_{fold_number}"
    )

    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        fold_directory
        / f"hvnet_fold{fold_number}_best.pth"
    )

    history = []

    best_validation_accuracy = (
        -float("inf")
    )

    best_epoch = None
    best_metrics = None

    # ==============================================================
    # Epoch loop
    # ==============================================================

    for epoch_index in range(
        epochs
    ):

        epoch_number = (
            epoch_index
            + 1
        )

        # ----------------------------------------------------------
        # Epochs 1--5:
        # backbone frozen
        #
        # Epoch 6:
        # unfreeze backbone
        # ----------------------------------------------------------

        if (
            freeze_epochs > 0
            and epoch_number
            == freeze_epochs + 1
        ):

            model.unfreeze_backbone()

            print(
                f"\nEpoch {epoch_number}: "
                "visual backbone unfrozen."
            )

        # ----------------------------------------------------------
        # Training
        # ----------------------------------------------------------

        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_max_norm=
                gradient_clip,
        )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        (
            val_losses,
            val_metrics,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        # ----------------------------------------------------------
        # Record current learning rates before scheduler step.
        # ----------------------------------------------------------

        current_lrs = [
            float(
                group[
                    "lr"
                ]
            )
            for group
            in optimizer.param_groups
        ]

        # ----------------------------------------------------------
        # Cosine annealing
        # ----------------------------------------------------------

        scheduler.step()

        record = {
            "epoch":
                epoch_number,

            "train":
                train_losses,

            "validation_loss":
                val_losses,

            "validation_metrics":
                {
                    key:
                        float(value)

                    for key, value
                    in val_metrics.items()

                    if np.isscalar(
                        value
                    )
                },

            "learning_rates":
                current_lrs,
        }

        history.append(
            record
        )

        # ----------------------------------------------------------
        # Checkpoint selection:
        #
        # best validation accuracy
        # ----------------------------------------------------------

        validation_accuracy = (
            val_metrics[
                "accuracy"
            ]
        )

        is_best = (
            validation_accuracy
            > best_validation_accuracy
        )

        if is_best:

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = (
                epoch_number
            )

            best_metrics = {
                key:
                    float(value)

                for key, value
                in val_metrics.items()

                if np.isscalar(
                    value
                )
            }

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_number,
                fold=fold_number,
                validation_metrics=
                    best_metrics,
                config=config,
            )

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------

        print(
            f"Fold {fold_number:02d} | "
            f"Epoch {epoch_number:03d}/{epochs:03d} | "
            f"Train Loss "
            f"{train_losses['total_loss']:.4f} | "
            f"Val Loss "
            f"{val_losses['total_loss']:.4f} | "
            f"Val Acc "
            f"{val_metrics['accuracy']:.2f} | "
            f"Val Spec "
            f"{val_metrics['specificity']:.2f} | "
            f"Val AUC "
            f"{val_metrics['auc']:.2f} | "
            f"Val Kappa "
            f"{val_metrics['kappa']:.2f}"
            + (
                " | BEST"
                if is_best
                else ""
            )
        )

    # ==============================================================
    # Save fold history
    # ==============================================================

    history_path = (
        fold_directory
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    fold_summary = {
        "fold":
            fold_number,

        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            best_validation_accuracy,

        "best_validation_metrics":
            best_metrics,

        "checkpoint":
            str(
                checkpoint_path
            ),
    }

    summary_path = (
        fold_directory
        / "fold_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            fold_summary,
            file,
            indent=2,
        )

    print(
        f"\nFold {fold_number} complete."
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    return fold_summary


# ======================================================================
# Command-line arguments
# ======================================================================


def parse_args():
    """
    Command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=
            "Train HVNet using patient-level five-fold "
            "cross-validation.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Internal cohort metadata CSV. "
            "Overrides data.metadata_file in YAML."
        ),
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=(
            "Root directory containing CFP images. "
            "Overrides data.root_dir in YAML."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory for fold-specific outputs.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--split-column",
        type=str,
        default="split",
        help=(
            "Metadata column containing the fixed "
            "internal cohort assignment."
        ),
    )

    parser.add_argument(
        "--development-value",
        type=str,
        default="development",
        help=(
            "Value identifying development samples "
            "in the split column."
        ),
    )

    parser.add_argument(
        "--internal-test-value",
        type=str,
        default="internal_test",
        help=(
            "Value identifying fixed internal-test "
            "samples."
        ),
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Train only one fold (1--5). "
            "By default all five folds are trained."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================


def main():

    args = parse_args()

    config = load_config(
        args.config
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    experiment_cfg = config.get(
        "experiment",
        {},
    )

    data_cfg = config.get(
        "data",
        {},
    )

    training_cfg = config.get(
        "training",
        {},
    )

    preprocessing_cfg = config.get(
        "preprocessing",
        {},
    )

    augmentation_cfg = config.get(
        "augmentation",
        {},
    )

    seed = experiment_cfg.get(
        "seed",
        42,
    )

    num_folds = (
        training_cfg
        .get(
            "cross_validation",
            {},
        )
        .get(
            "num_folds",
            5,
        )
    )

    set_random_seed(
        seed
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    requested_device = (
        training_cfg.get(
            "device",
            "cuda",
        )
    )

    if (
        requested_device
        == "cuda"
        and torch.cuda.is_available()
    ):

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------

    metadata_file = (
        args.metadata
        if args.metadata is not None
        else data_cfg.get(
            "metadata_file"
        )
    )

    root_dir = (
        args.root_dir
        if args.root_dir is not None
        else data_cfg.get(
            "root_dir"
        )
    )

    if metadata_file is None:
        raise ValueError(
            "No metadata CSV was provided. "
            "Set data.metadata_file in configs/default.yaml "
            "or use --metadata."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------------

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

    train_aug_cfg = (
        augmentation_cfg.get(
            "train",
            {},
        )
    )

    crop_scale = (
        train_aug_cfg
        .get(
            "random_resized_crop",
            {},
        )
        .get(
            "scale",
            [
                0.70,
                1.00,
            ],
        )
    )

    train_transform = (
        build_train_transform(
            image_size=image_size,
            crop_scale=tuple(
                crop_scale
            ),
        )
    )

    val_transform = (
        build_eval_transform(
            image_size=image_size,
        )
    )

    # --------------------------------------------------------------
    # Dataset
    #
    # Two dataset instances are used so that training samples receive
    # augmentation while validation samples receive deterministic
    # preprocessing.
    # --------------------------------------------------------------

    common_dataset_kwargs = {
        "metadata_file":
            metadata_file,

        "root_dir":
            root_dir,

        "image_column":
            data_cfg.get(
                "image_column",
                "image_path",
            ),

        "patient_id_column":
            data_cfg.get(
                "patient_id_column",
                "patient_id",
            ),

        "iop_column":
            "iop",

        "acd_column":
            "acd",

        "cag_column":
            "cag",

        "label_column":
            data_cfg.get(
                "label_column",
                "label",
            ),
    }

    train_dataset = HVNetDataset(
        transform=train_transform,
        **common_dataset_kwargs,
    )

    val_dataset = HVNetDataset(
        transform=val_transform,
        **common_dataset_kwargs,
    )

    metadata = (
        train_dataset
        .get_metadata()
    )

    patient_id_column = (
        train_dataset
        .patient_id_column
    )

    label_column = (
        train_dataset
        .label_column
    )

    # --------------------------------------------------------------
    # Fixed development/internal-test partition
    # --------------------------------------------------------------

    (
        development_indices,
        internal_test_indices,
    ) = resolve_internal_split(
        metadata=metadata,
        patient_id_column=
            patient_id_column,
        label_column=
            label_column,
        split_column=
            args.split_column,
        development_value=
            args.development_value,
        internal_test_value=
            args.internal_test_value,
        seed=seed,
        output_dir=output_dir,
    )

    print(
        "\nInternal cohort:"
    )

    print(
        f"Total eyes: "
        f"{len(metadata)}"
    )

    print(
        f"Development eyes: "
        f"{len(development_indices)}"
    )

    print(
        f"Fixed internal-test eyes: "
        f"{len(internal_test_indices)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The fixed internal-test subset will NOT be used "
        "during training or checkpoint selection."
    )

    # --------------------------------------------------------------
    # Build 5 folds within development only
    # --------------------------------------------------------------

    folds = build_development_folds(
        metadata=metadata,
        development_indices=
            development_indices,
        patient_id_column=
            patient_id_column,
        label_column=
            label_column,
        num_folds=num_folds,
        seed=seed,
    )

    # --------------------------------------------------------------
    # Save fold assignments for reproducibility
    # --------------------------------------------------------------

    fold_assignment = (
        metadata.copy()
    )

    fold_assignment[
        "training_role"
    ] = "internal_test"

    fold_assignment.loc[
        development_indices,
        "training_role",
    ] = "development"

    fold_assignment[
        "cv_fold"
    ] = np.nan

    for (
        fold_index,
        (_, val_indices),
    ) in enumerate(
        folds,
        start=1,
    ):

        fold_assignment.loc[
            val_indices,
            "cv_fold",
        ] = fold_index

    fold_assignment.to_csv(
        output_dir
        / "cv_assignments.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Select folds
    # --------------------------------------------------------------

    if args.fold is not None:

        if not (
            1
            <= args.fold
            <= num_folds
        ):

            raise ValueError(
                f"--fold must be between 1 and {num_folds}."
            )

        fold_numbers = [
            args.fold
        ]

    else:

        fold_numbers = list(
            range(
                1,
                num_folds + 1,
            )
        )

    # ==============================================================
    # Train folds
    # ==============================================================

    summaries = []

    for fold_number in fold_numbers:

        train_indices, val_indices = (
            folds[
                fold_number - 1
            ]
        )

        fold_summary = train_fold(
            fold_number=fold_number,
            train_indices=train_indices,
            val_indices=val_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            output_dir=output_dir,
            device=device,
            num_workers=
                args.num_workers,
            seed=seed,
        )

        summaries.append(
            fold_summary
        )

    # --------------------------------------------------------------
    # Save global training summary
    # --------------------------------------------------------------

    overall_summary = {
        "num_folds_trained":
            len(summaries),

        "folds":
            summaries,

        "development_eyes":
            int(
                len(
                    development_indices
                )
            ),

        "fixed_internal_test_eyes":
            int(
                len(
                    internal_test_indices
                )
            ),

        "seed":
            seed,
    }

    with open(
        output_dir
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            overall_summary,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HVNet training complete."
    )

    print(
        "=" * 70
    )

    print(
        f"Outputs saved to: {output_dir}"
    )

    print(
        "\nThe fixed internal-test and held-out cohorts "
        "should now be evaluated using evaluate.py."
    )


if __name__ == "__main__":
    main()
