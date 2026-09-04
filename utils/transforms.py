"""
Image preprocessing and augmentation utilities for HVNet.

This module implements the CFP preprocessing protocol described in the
HVNet manuscript.

Training preprocessing
----------------------

The training pipeline includes:

    1. RandomResizedCrop to 224 x 224
       scale = [0.70, 1.00]

    2. Random horizontal flipping

    3. ColorJitter

    4. RandAugment

    5. Conversion to tensor

    6. ImageNet normalization

    7. RandomErasing


Validation / test preprocessing
-------------------------------

Validation and test images are processed deterministically using:

    1. Resize to 224 x 224

    2. Conversion to tensor

    3. ImageNet normalization


Important
---------

The manuscript explicitly specifies:

    - image size = 224 x 224
    - RandomResizedCrop scale = [0.70, 1.00]
    - horizontal flipping
    - ColorJitter
    - RandAugment
    - RandomErasing
    - ImageNet normalization

The exact numerical strengths/probabilities of ColorJitter,
RandAugment, horizontal flipping, and RandomErasing should match the
original experimental implementation before the repository is finalized.

The current defaults for parameters not explicitly specified in the
manuscript are operational defaults and can be overwritten through the
function arguments.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torchvision import transforms


# ======================================================================
# ImageNet normalization
# ======================================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ======================================================================
# Default manuscript-level parameters
# ======================================================================

DEFAULT_IMAGE_SIZE = 224

# Explicitly reported in the manuscript.
DEFAULT_RANDOM_CROP_SCALE = (
    0.70,
    1.00,
)


# ======================================================================
# Training transform
# ======================================================================


def build_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_scale: Tuple[float, float] = DEFAULT_RANDOM_CROP_SCALE,

    horizontal_flip_prob: float = 0.5,

    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.05,

    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,

    random_erasing_prob: float = 0.25,
    random_erasing_scale: Tuple[float, float] = (0.02, 0.20),
    random_erasing_ratio: Tuple[float, float] = (
        0.3,
        3.3,
    ),

    imagenet_mean: Sequence[float] = IMAGENET_MEAN,
    imagenet_std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Build the training-time CFP preprocessing pipeline.

    Parameters
    ----------
    image_size : int
        Final image size.

        Manuscript configuration:
            224

    crop_scale : tuple of float
        Scale range used by RandomResizedCrop.

        Manuscript configuration:
            (0.70, 1.00)

    horizontal_flip_prob : float
        Probability of horizontal flipping.

        The manuscript reports horizontal flipping but does not
        explicitly specify the probability in the current Methods text.

    color_jitter_brightness : float
        Brightness strength for ColorJitter.

    color_jitter_contrast : float
        Contrast strength for ColorJitter.

    color_jitter_saturation : float
        Saturation strength for ColorJitter.

    color_jitter_hue : float
        Hue strength for ColorJitter.

    randaugment_num_ops : int
        Number of RandAugment operations.

    randaugment_magnitude : int
        RandAugment magnitude.

    random_erasing_prob : float
        Probability of RandomErasing.

    random_erasing_scale : tuple
        Area scale used by RandomErasing.

    random_erasing_ratio : tuple
        Aspect-ratio range used by RandomErasing.

    imagenet_mean : sequence
        ImageNet normalization mean.

    imagenet_std : sequence
        ImageNet normalization standard deviation.

    Returns
    -------
    torchvision.transforms.Compose
        Training transform pipeline.
    """

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than zero."
        )

    if (
        len(crop_scale) != 2
        or crop_scale[0] <= 0
        or crop_scale[1] <= 0
        or crop_scale[0] > crop_scale[1]
    ):
        raise ValueError(
            "crop_scale must be a valid (min, max) tuple."
        )

    if not 0.0 <= horizontal_flip_prob <= 1.0:
        raise ValueError(
            "horizontal_flip_prob must lie in [0, 1]."
        )

    if not 0.0 <= random_erasing_prob <= 1.0:
        raise ValueError(
            "random_erasing_prob must lie in [0, 1]."
        )

    # --------------------------------------------------------------
    # Training augmentation
    # --------------------------------------------------------------

    train_transform = transforms.Compose(
        [
            # ======================================================
            # Random resized crop
            #
            # Manuscript:
            #
            # scale = 0.7 -- 1.0
            # output = 224 x 224
            # ======================================================

            transforms.RandomResizedCrop(
                size=(
                    image_size,
                    image_size,
                ),
                scale=crop_scale,
            ),

            # ======================================================
            # Horizontal flip
            # ======================================================

            transforms.RandomHorizontalFlip(
                p=horizontal_flip_prob
            ),

            # ======================================================
            # Color jitter
            # ======================================================

            transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            ),

            # ======================================================
            # RandAugment
            #
            # Applied while the image is still in PIL format.
            # ======================================================

            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            ),

            # ======================================================
            # PIL -> Tensor
            #
            # [H, W, C]
            #       ->
            # [C, H, W]
            #
            # pixel range:
            # 0--255 -> 0--1
            # ======================================================

            transforms.ToTensor(),

            # ======================================================
            # ImageNet normalization
            # ======================================================

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std,
            ),

            # ======================================================
            # Random erasing
            #
            # Must be applied after conversion to tensor.
            # ======================================================

            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=0,
            ),
        ]
    )

    return train_transform


# ======================================================================
# Validation / test transform
# ======================================================================


def build_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    imagenet_mean: Sequence[float] = IMAGENET_MEAN,
    imagenet_std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Build deterministic preprocessing for validation and testing.

    No random augmentation is performed.

    Parameters
    ----------
    image_size : int
        Final image size.

        Manuscript configuration:
            224

    imagenet_mean : sequence
        ImageNet normalization mean.

    imagenet_std : sequence
        ImageNet normalization standard deviation.

    Returns
    -------
    torchvision.transforms.Compose
        Validation / test transform pipeline.
    """

    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than zero."
        )

    eval_transform = transforms.Compose(
        [
            # ======================================================
            # Resize
            # ======================================================

            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            # ======================================================
            # Convert to tensor
            # ======================================================

            transforms.ToTensor(),

            # ======================================================
            # ImageNet normalization
            # ======================================================

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        ]
    )

    return eval_transform


# ======================================================================
# Test transform
# ======================================================================


def build_test_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    imagenet_mean: Sequence[float] = IMAGENET_MEAN,
    imagenet_std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Alias for deterministic test-time preprocessing.

    The test pipeline is intentionally identical to validation.
    """

    return build_eval_transform(
        image_size=image_size,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
    )


# ======================================================================
# Build all transforms
# ======================================================================


def build_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_scale: Tuple[float, float] = DEFAULT_RANDOM_CROP_SCALE,

    horizontal_flip_prob: float = 0.5,

    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.05,

    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,

    random_erasing_prob: float = 0.25,
) -> Dict[str, transforms.Compose]:
    """
    Convenience function returning train, validation, and test transforms.

    Returns
    -------
    dict

        {
            "train": train_transform,
            "val": validation_transform,
            "test": test_transform,
        }
    """

    train_transform = build_train_transform(
        image_size=image_size,
        crop_scale=crop_scale,

        horizontal_flip_prob=
            horizontal_flip_prob,

        color_jitter_brightness=
            color_jitter_brightness,

        color_jitter_contrast=
            color_jitter_contrast,

        color_jitter_saturation=
            color_jitter_saturation,

        color_jitter_hue=
            color_jitter_hue,

        randaugment_num_ops=
            randaugment_num_ops,

        randaugment_magnitude=
            randaugment_magnitude,

        random_erasing_prob=
            random_erasing_prob,
    )

    validation_transform = build_eval_transform(
        image_size=image_size,
    )

    test_transform = build_test_transform(
        image_size=image_size,
    )

    return {
        "train": train_transform,
        "val": validation_transform,
        "test": test_transform,
    }


# ======================================================================
# Denormalization utility
# ======================================================================


def denormalize_imagenet(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Reverse ImageNet normalization.

    Useful for visualization, Grad-CAM overlays, or debugging.

    Parameters
    ----------
    image : torch.Tensor

        Supported shapes:

            [3, H, W]

        or:

            [B, 3, H, W]

    Returns
    -------
    torch.Tensor
        Image tensor approximately restored to the [0, 1] range.
    """

    if image.ndim not in (
        3,
        4,
    ):
        raise ValueError(
            "image must have shape [3, H, W] "
            "or [B, 3, H, W]."
        )

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=image.dtype,
        device=image.device,
    )

    std = torch.tensor(
        IMAGENET_STD,
        dtype=image.dtype,
        device=image.device,
    )

    if image.ndim == 3:

        mean = mean.view(
            3,
            1,
            1,
        )

        std = std.view(
            3,
            1,
            1,
        )

    else:

        mean = mean.view(
            1,
            3,
            1,
            1,
        )

        std = std.view(
            1,
            3,
            1,
            1,
        )

    restored = (
        image * std
        + mean
    )

    restored = restored.clamp(
        0.0,
        1.0,
    )

    return restored


# ======================================================================
# Short aliases
# ======================================================================

get_train_transform = build_train_transform
get_val_transform = build_eval_transform
get_test_transform = build_test_transform


# ======================================================================
# Minimal sanity check
# ======================================================================


if __name__ == "__main__":

    from PIL import Image

    # --------------------------------------------------------------
    # Artificial RGB image.
    #
    # This block does not access clinical data.
    # --------------------------------------------------------------

    example_image = Image.new(
        mode="RGB",
        size=(
            512,
            512,
        ),
        color=(
            128,
            128,
            128,
        ),
    )

    transforms_dict = build_transforms(
        image_size=224,
        crop_scale=(
            0.70,
            1.00,
        ),
    )

    training_image = transforms_dict[
        "train"
    ](
        example_image
    )

    validation_image = transforms_dict[
        "val"
    ](
        example_image
    )

    test_image = transforms_dict[
        "test"
    ](
        example_image
    )

    print(
        "Training image shape:",
        training_image.shape,
    )

    print(
        "Validation image shape:",
        validation_image.shape,
    )

    print(
        "Test image shape:",
        test_image.shape,
    )

    restored = denormalize_imagenet(
        validation_image
    )

    print(
        "Denormalized range:",
        float(restored.min()),
        float(restored.max()),
    )
