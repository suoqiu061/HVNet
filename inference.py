"""
Single-sample inference for HVNet.

HVNet:
    Hypothesis-and-Verify Network for three-class glaucoma triage.

Input
-----

Each inference sample contains:

    1. One color fundus photograph (CFP)
    2. IOP status
    3. SLE-derived ACD status
    4. Van Herick-based CAG

Clinical encoding
-----------------

IOP:

    0 = not elevated
    1 = elevated (> 21 mmHg)

ACD:

    0 = normal
    1 = shallow

CAG:

    integer grade from 0 to 4

The structured clinical branch may be labeled "Text" in schematic figures,
but no free-text or natural-language input is used by HVNet.

Output
------

The script reports:

    - final predicted class
    - final class probabilities
    - clinical-branch probabilities
    - visual-branch probabilities
    - Dempster--Shafer fused probabilities
    - sample-adaptive gate weights
    - clinical uncertainty
    - visual uncertainty
    - DS uncertainty
    - cross-modal DS conflict

Important
---------

HVNet is intended for research use and subtype-oriented triage research.
The output should not be interpreted as an autonomous clinical diagnosis
or treatment recommendation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from PIL import Image

from models.hvnet import HVNet
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
# Model construction
# ======================================================================


def build_model_from_config(
    config: Dict,
) -> HVNet:
    """
    Construct HVNet using the architecture specified by the configuration.

    ImageNet weights are not loaded here because inference restores the
    complete trained model state from the HVNet checkpoint.
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

    local_cfg = visual_cfg.get(
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
        # ConvNeXt visual backbone
        # ----------------------------------------------------------

        backbone_name=backbone_cfg.get(
            "name",
            "convnext_tiny",
        ),

        pretrained=False,

        backbone_out_indices=tuple(
            backbone_cfg.get(
                "out_indices",
                [1, 2],
            )
        ),

        # ----------------------------------------------------------
        # Clinical Hypothesis Encoder
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
            local_cfg.get(
                "num_local_patches",
                4,
            ),

        patch_size=
            local_cfg.get(
                "patch_size",
                3,
            ),

        local_transformer_heads=
            local_cfg.get(
                "transformer_heads",
                4,
            ),

        local_transformer_layers=
            local_cfg.get(
                "transformer_layers",
                1,
            ),

        local_feedforward_dim=
            local_cfg.get(
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
# Checkpoint loading
# ======================================================================


def load_checkpoint(
    checkpoint_path: str,
    fallback_config: Dict,
    device: torch.device,
) -> Tuple[
    HVNet,
    Dict,
    Dict,
]:
    """
    Load an HVNet checkpoint.

    If the checkpoint contains the configuration used during training,
    that configuration is preferred over the externally supplied YAML.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Unexpected checkpoint format."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model_config = checkpoint.get(
        "config",
        fallback_config,
    )

    model = build_model_from_config(
        model_config
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
        model_config,
    )


# ======================================================================
# Input validation
# ======================================================================


def validate_clinical_inputs(
    iop: int,
    acd: int,
    cag: int,
) -> None:
    """
    Validate the three structured clinical indicators.
    """

    if iop not in (
        0,
        1,
    ):
        raise ValueError(
            "IOP must be encoded as 0 or 1."
        )

    if acd not in (
        0,
        1,
    ):
        raise ValueError(
            "ACD must be encoded as 0 or 1."
        )

    if cag not in (
        0,
        1,
        2,
        3,
        4,
    ):
        raise ValueError(
            "CAG must be an integer from 0 to 4."
        )


# ======================================================================
# CFP preprocessing
# ======================================================================


def preprocess_image(
    image_path: str,
    config: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and preprocess one CFP image.

    Evaluation-time preprocessing matches the manuscript:

        Resize to 224 x 224
        ToTensor
        ImageNet normalization
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"CFP image not found: {image_path}"
        )

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

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image_tensor = transform(
        image
    )

    # --------------------------------------------------------------
    # [3, H, W]
    # ->
    # [1, 3, H, W]
    # --------------------------------------------------------------

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(
            device
        )
    )

    return image_tensor


# ======================================================================
# Utility
# ======================================================================


def tensor_scalar(
    tensor: torch.Tensor,
) -> float:
    """
    Convert a single-value tensor to Python float.
    """

    return float(
        tensor
        .detach()
        .cpu()
        .item()
    )


def tensor_vector(
    tensor: torch.Tensor,
) -> Sequence[float]:
    """
    Convert a single-sample probability vector into a Python list.
    """

    return (
        tensor
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )


# ======================================================================
# Single-sample inference
# ======================================================================


@torch.no_grad()
def infer_single_sample(
    model: HVNet,
    image_tensor: torch.Tensor,
    iop: int,
    acd: int,
    cag: int,
    class_names: Sequence[str],
    device: torch.device,
) -> Dict:
    """
    Perform HVNet inference on one CFP + clinical-indicator sample.
    """

    validate_clinical_inputs(
        iop=iop,
        acd=acd,
        cag=cag,
    )

    # --------------------------------------------------------------
    # Structured clinical indicators
    #
    # Shape:
    #     [1]
    #
    # LongTensor is required because clinical_encoder.py uses
    # embedding layers.
    # --------------------------------------------------------------

    iop_tensor = torch.tensor(
        [
            iop
        ],
        dtype=torch.long,
        device=device,
    )

    acd_tensor = torch.tensor(
        [
            acd
        ],
        dtype=torch.long,
        device=device,
    )

    cag_tensor = torch.tensor(
        [
            cag
        ],
        dtype=torch.long,
        device=device,
    )

    # ==============================================================
    # HVNet
    # ==============================================================

    outputs = model(
        image=image_tensor,
        iop=iop_tensor,
        acd=acd_tensor,
        cag=cag_tensor,
        return_auxiliary=True,
    )

    # --------------------------------------------------------------
    # Final probabilities
    # --------------------------------------------------------------

    p_final = outputs[
        "p_final"
    ][0]

    predicted_index = int(
        torch.argmax(
            p_final
        ).item()
    )

    if (
        predicted_index
        < len(
            class_names
        )
    ):

        predicted_class = (
            class_names[
                predicted_index
            ]
        )

    else:

        predicted_class = (
            str(
                predicted_index
            )
        )

    # ==============================================================
    # Probability dictionaries
    # ==============================================================

    def probability_dict(
        probability_tensor: torch.Tensor,
    ) -> Dict[str, float]:

        values = tensor_vector(
            probability_tensor[
                0
            ]
        )

        return {
            str(
                class_names[index]
            ):
                float(
                    values[
                        index
                    ]
                )

            for index in range(
                min(
                    len(
                        values
                    ),
                    len(
                        class_names
                    ),
                )
            )
        }

    final_probabilities = (
        probability_dict(
            outputs[
                "p_final"
            ]
        )
    )

    clinical_probabilities = (
        probability_dict(
            outputs[
                "p_clin"
            ]
        )
    )

    visual_probabilities = (
        probability_dict(
            outputs[
                "p_vis"
            ]
        )
    )

    ds_probabilities = (
        probability_dict(
            outputs[
                "p_ds"
            ]
        )
    )

    # ==============================================================
    # Gate weights
    # ==============================================================

    gate_weights = {
        "clinical":
            tensor_scalar(
                outputs[
                    "w_clin"
                ][
                    0
                ]
            ),

        "visual":
            tensor_scalar(
                outputs[
                    "w_vis"
                ][
                    0
                ]
            ),

        "ds":
            tensor_scalar(
                outputs[
                    "w_ds"
                ][
                    0
                ]
            ),
    }

    # ==============================================================
    # Uncertainty
    # ==============================================================

    uncertainty = {
        "clinical":
            tensor_scalar(
                outputs[
                    "u_clin"
                ][
                    0
                ]
            ),

        "visual":
            tensor_scalar(
                outputs[
                    "u_vis"
                ][
                    0
                ]
            ),
    }

    if (
        "u_ds"
        in outputs
    ):

        uncertainty[
            "ds"
        ] = tensor_scalar(
            outputs[
                "u_ds"
            ][
                0
            ]
        )

    # ==============================================================
    # DS conflict
    # ==============================================================

    ds_conflict = tensor_scalar(
        outputs[
            "conflict"
        ][
            0
        ]
    )

    # ==============================================================
    # Result
    # ==============================================================

    result = {
        "prediction": {
            "class_index":
                predicted_index,

            "class_name":
                predicted_class,

            "confidence":
                float(
                    p_final[
                        predicted_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),
        },

        "probabilities": {
            "final":
                final_probabilities,

            "clinical":
                clinical_probabilities,

            "visual":
                visual_probabilities,

            "ds":
                ds_probabilities,
        },

        "gate_weights":
            gate_weights,

        "uncertainty":
            uncertainty,

        "ds_conflict":
            ds_conflict,

        "clinical_input": {
            "iop":
                int(
                    iop
                ),

            "acd":
                int(
                    acd
                ),

            "cag":
                int(
                    cag
                ),
        },
    }

    return result


# ======================================================================
# Console formatting
# ======================================================================


def print_inference_result(
    result: Dict,
) -> None:
    """
    Print inference result in human-readable form.
    """

    prediction = result[
        "prediction"
    ]

    probabilities = result[
        "probabilities"
    ]

    gate_weights = result[
        "gate_weights"
    ]

    uncertainty = result[
        "uncertainty"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HVNet Inference"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal prediction:"
    )

    print(
        f"  Class:      "
        f"{prediction['class_name']}"
    )

    print(
        f"  Confidence: "
        f"{prediction['confidence']:.4f}"
    )

    # --------------------------------------------------------------
    # Final probabilities
    # --------------------------------------------------------------

    print(
        "\nFinal class probabilities:"
    )

    for (
        class_name,
        probability,
    ) in probabilities[
        "final"
    ].items():

        print(
            f"  {class_name:10s}: "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------------
    # Branch probabilities
    # --------------------------------------------------------------

    print(
        "\nClinical branch:"
    )

    for (
        class_name,
        probability,
    ) in probabilities[
        "clinical"
    ].items():

        print(
            f"  {class_name:10s}: "
            f"{probability:.4f}"
        )

    print(
        "\nVisual branch:"
    )

    for (
        class_name,
        probability,
    ) in probabilities[
        "visual"
    ].items():

        print(
            f"  {class_name:10s}: "
            f"{probability:.4f}"
        )

    print(
        "\nDS fusion:"
    )

    for (
        class_name,
        probability,
    ) in probabilities[
        "ds"
    ].items():

        print(
            f"  {class_name:10s}: "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------------
    # Gate
    # --------------------------------------------------------------

    print(
        "\nSample-adaptive gate weights:"
    )

    print(
        f"  Clinical: {gate_weights['clinical']:.4f}"
    )

    print(
        f"  Visual:   {gate_weights['visual']:.4f}"
    )

    print(
        f"  DS:       {gate_weights['ds']:.4f}"
    )

    print(
        f"  Sum:      "
        f"{sum(gate_weights.values()):.4f}"
    )

    # --------------------------------------------------------------
    # Evidential uncertainty
    # --------------------------------------------------------------

    print(
        "\nEvidential uncertainty:"
    )

    print(
        f"  Clinical: "
        f"{uncertainty['clinical']:.4f}"
    )

    print(
        f"  Visual:   "
        f"{uncertainty['visual']:.4f}"
    )

    if (
        "ds"
        in uncertainty
    ):

        print(
            f"  DS:       "
            f"{uncertainty['ds']:.4f}"
        )

    print(
        "\nDS conflict:"
    )

    print(
        f"  {result['ds_conflict']:.4f}"
    )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Research use only. "
        "This output is not an autonomous clinical diagnosis."
    )


# ======================================================================
# Command-line arguments
# ======================================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run single-sample inference using HVNet."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to HVNet configuration YAML.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained HVNet checkpoint.",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the CFP image.",
    )

    parser.add_argument(
        "--iop",
        type=int,
        required=True,
        choices=[
            0,
            1,
        ],
        help=(
            "Binary IOP status: "
            "0 = not elevated, "
            "1 = elevated (>21 mmHg)."
        ),
    )

    parser.add_argument(
        "--acd",
        type=int,
        required=True,
        choices=[
            0,
            1,
        ],
        help=(
            "Binary SLE-derived ACD: "
            "0 = normal, "
            "1 = shallow."
        ),
    )

    parser.add_argument(
        "--cag",
        type=int,
        required=True,
        choices=[
            0,
            1,
            2,
            3,
            4,
        ],
        help="Van Herick CAG from 0 to 4.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Inference device. "
            "Defaults to CUDA when available."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional path for saving the "
            "inference result as JSON."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================


def main():

    args = parse_args()

    # ==============================================================
    # Device
    # ==============================================================

    if args.device is None:

        if torch.cuda.is_available():

            device = torch.device(
                "cuda"
            )

        else:

            device = torch.device(
                "cpu"
            )

    else:

        if (
            args.device
            == "cuda"
            and not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device(
            args.device
        )

    print(
        "Device:",
        device,
    )

    # ==============================================================
    # Config
    # ==============================================================

    config = load_config(
        args.config
    )

    # ==============================================================
    # Checkpoint
    # ==============================================================

    (
        model,
        checkpoint,
        model_config,
    ) = load_checkpoint(
        checkpoint_path=
            args.checkpoint,

        fallback_config=
            config,

        device=
            device,
    )

    # --------------------------------------------------------------
    # Prefer class names stored in the training configuration.
    # --------------------------------------------------------------

    experiment_cfg = (
        model_config.get(
            "experiment",
            {},
        )
    )

    class_names = (
        experiment_cfg.get(
            "class_names",
            [
                "Normal",
                "OAG",
                "ACG",
            ],
        )
    )

    # ==============================================================
    # CFP preprocessing
    # ==============================================================

    image_tensor = preprocess_image(
        image_path=
            args.image,

        config=
            model_config,

        device=
            device,
    )

    # ==============================================================
    # Inference
    # ==============================================================

    result = infer_single_sample(
        model=model,
        image_tensor=
            image_tensor,

        iop=
            args.iop,

        acd=
            args.acd,

        cag=
            args.cag,

        class_names=
            class_names,

        device=
            device,
    )

    # --------------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------------

    result[
        "image"
    ] = str(
        Path(
            args.image
        )
    )

    result[
        "checkpoint"
    ] = str(
        Path(
            args.checkpoint
        )
    )

    if (
        "fold"
        in checkpoint
    ):

        result[
            "checkpoint_fold"
        ] = checkpoint[
            "fold"
        ]

    if (
        "epoch"
        in checkpoint
    ):

        result[
            "checkpoint_epoch"
        ] = checkpoint[
            "epoch"
        ]

    # ==============================================================
    # Console
    # ==============================================================

    print_inference_result(
        result
    )

    # ==============================================================
    # Optional JSON
    # ==============================================================

    if (
        args.output_json
        is not None
    ):

        output_path = Path(
            args.output_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
            )

        print(
            "\nInference result saved to:"
        )

        print(
            output_path
        )


if __name__ == "__main__":
    main()
