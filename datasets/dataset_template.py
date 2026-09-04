"""
Dataset template for HVNet.

Each sample used by HVNet contains:

    1. One color fundus photograph (CFP)
    2. Three structured clinical indicators:
        - IOP
        - ACD
        - CAG
    3. One three-class reference label
    4. A patient identifier used only for patient-level partitioning

The model input is:

    image
    iop
    acd
    cag

The patient identifier is NOT used as a predictive feature.

----------------------------------------------------------------------
Clinical encoding
----------------------------------------------------------------------

IOP
    Binary indicator:

        0 = not elevated
        1 = elevated (> 21 mmHg)

ACD
    Binary SLE-derived anterior chamber depth:

        0 = normal
        1 = shallow

CAG
    Van Herick-based chamber-angle grade:

        0, 1, 2, 3, or 4

----------------------------------------------------------------------
Expected metadata format
----------------------------------------------------------------------

A CSV file may follow the structure:

    patient_id,image_path,iop,acd,cag,label

For example:

    demo_001,images/demo_001.jpg,0,0,4,0
    demo_002,images/demo_002.jpg,1,1,1,2

The exact class-to-index mapping should match the mapping used in the
original HVNet experiments.

No patient-identifiable or protected clinical information should be
included in the public repository.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HVNetDataset(Dataset):
    """
    Dataset for CFP + structured clinical indicators.

    Parameters
    ----------
    metadata_file : str or Path
        Path to the CSV metadata file.

    root_dir : str or Path, optional
        Root directory containing CFP images.

        If ``image_path`` values in the metadata file are relative,
        they are resolved relative to ``root_dir``.

        If they are already absolute paths, ``root_dir`` is ignored
        for those samples.

    transform : callable, optional
        Image preprocessing / augmentation function.

        It should accept a PIL image and return a tensor.

        Example:

            transform(image) -> torch.Tensor [3, H, W]

    image_column : str
        Metadata column containing CFP paths.

    patient_id_column : str
        Metadata column containing patient identifiers.

    iop_column : str
        Metadata column containing binary IOP status.

    acd_column : str
        Metadata column containing binary ACD status.

    cag_column : str
        Metadata column containing Van Herick CAG.

    label_column : str
        Metadata column containing the three-class target label.

    class_to_index : dict, optional
        Optional mapping for string labels.

        For example, only if this matches the original experiment:

            {
                "Normal": 0,
                "OAG": 1,
                "ACG": 2,
            }

        If None, labels must already be integer encoded.

    validate_data : bool
        Whether to validate metadata values during initialization.
    """

    REQUIRED_DEFAULT_COLUMNS = [
        "patient_id",
        "image_path",
        "iop",
        "acd",
        "cag",
        "label",
    ]

    def __init__(
        self,
        metadata_file: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        image_column: str = "image_path",
        patient_id_column: str = "patient_id",
        iop_column: str = "iop",
        acd_column: str = "acd",
        cag_column: str = "cag",
        label_column: str = "label",
        class_to_index: Optional[Dict[str, int]] = None,
        validate_data: bool = True,
    ) -> None:

        super().__init__()

        self.metadata_file = Path(metadata_file)

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}"
            )

        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else None
        )

        self.transform = transform

        self.image_column = image_column
        self.patient_id_column = patient_id_column
        self.iop_column = iop_column
        self.acd_column = acd_column
        self.cag_column = cag_column
        self.label_column = label_column

        self.class_to_index = class_to_index

        # ----------------------------------------------------------
        # Load metadata
        # ----------------------------------------------------------

        self.metadata = pd.read_csv(
            self.metadata_file
        )

        if len(self.metadata) == 0:
            raise ValueError(
                f"Metadata file is empty: {self.metadata_file}"
            )

        # ----------------------------------------------------------
        # Verify required columns
        # ----------------------------------------------------------

        required_columns = [
            self.patient_id_column,
            self.image_column,
            self.iop_column,
            self.acd_column,
            self.cag_column,
            self.label_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required metadata columns: "
                + ", ".join(missing_columns)
            )

        # ----------------------------------------------------------
        # Encode labels if string mapping is provided
        # ----------------------------------------------------------

        if self.class_to_index is not None:
            self.metadata[self.label_column] = (
                self.metadata[self.label_column]
                .map(self.class_to_index)
            )

            if (
                self.metadata[self.label_column]
                .isna()
                .any()
            ):
                raise ValueError(
                    "At least one label could not be mapped using "
                    "class_to_index."
                )

        if validate_data:
            self._validate_metadata()

    # ==============================================================
    # Metadata validation
    # ==============================================================

    def _validate_metadata(self) -> None:
        """
        Validate structured clinical inputs and labels.
        """

        # ----------------------------------------------------------
        # Missing values
        # ----------------------------------------------------------

        required_columns = [
            self.patient_id_column,
            self.image_column,
            self.iop_column,
            self.acd_column,
            self.cag_column,
            self.label_column,
        ]

        for column in required_columns:

            if self.metadata[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values."
                )

        # ----------------------------------------------------------
        # IOP must be binary
        # ----------------------------------------------------------

        iop_values = set(
            self.metadata[self.iop_column]
            .astype(int)
            .unique()
            .tolist()
        )

        if not iop_values.issubset({0, 1}):
            raise ValueError(
                "IOP must be binary encoded as 0 or 1. "
                f"Observed values: {sorted(iop_values)}"
            )

        # ----------------------------------------------------------
        # ACD must be binary
        # ----------------------------------------------------------

        acd_values = set(
            self.metadata[self.acd_column]
            .astype(int)
            .unique()
            .tolist()
        )

        if not acd_values.issubset({0, 1}):
            raise ValueError(
                "ACD must be binary encoded as 0 or 1. "
                f"Observed values: {sorted(acd_values)}"
            )

        # ----------------------------------------------------------
        # CAG must be 0--4
        # ----------------------------------------------------------

        cag_values = set(
            self.metadata[self.cag_column]
            .astype(int)
            .unique()
            .tolist()
        )

        valid_cag_values = {
            0,
            1,
            2,
            3,
            4,
        }

        if not cag_values.issubset(
            valid_cag_values
        ):
            raise ValueError(
                "CAG must be an integer grade from 0 to 4. "
                f"Observed values: {sorted(cag_values)}"
            )

        # ----------------------------------------------------------
        # HVNet is a three-class task.
        #
        # We intentionally do not enforce a specific semantic
        # class-to-index mapping here.
        # ----------------------------------------------------------

        labels = (
            self.metadata[self.label_column]
            .astype(int)
        )

        unique_labels = sorted(
            labels.unique().tolist()
        )

        if len(unique_labels) > 3:
            raise ValueError(
                "HVNet is configured for a three-class task, "
                f"but found labels: {unique_labels}"
            )

        if min(unique_labels) < 0:
            raise ValueError(
                "Class labels must be non-negative integers."
            )

    # ==============================================================
    # Image path handling
    # ==============================================================

    def _resolve_image_path(
        self,
        image_path: Union[str, Path],
    ) -> Path:
        """
        Resolve an image path from the metadata table.
        """

        image_path = Path(
            str(image_path)
        )

        # Already absolute
        if image_path.is_absolute():
            return image_path

        # Resolve relative to root_dir
        if self.root_dir is not None:
            return (
                self.root_dir
                / image_path
            )

        # Otherwise resolve relative to the metadata CSV directory
        return (
            self.metadata_file.parent
            / image_path
        )

    # ==============================================================
    # Dataset interface
    # ==============================================================

    def __len__(self) -> int:
        """
        Return the number of eyes / samples.
        """

        return len(
            self.metadata
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:
        """
        Load one HVNet sample.

        Returns
        -------
        sample : dict

            image
                CFP tensor.

            iop
                Binary IOP tensor.

            acd
                Binary ACD tensor.

            cag
                Van Herick CAG tensor.

            label
                Target class tensor.

            patient_id
                Patient identifier used for dataset partitioning only.

            image_path
                Resolved image path for traceability/debugging.
        """

        row = self.metadata.iloc[
            index
        ]

        # ----------------------------------------------------------
        # CFP
        # ----------------------------------------------------------

        image_path = self._resolve_image_path(
            row[self.image_column]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"CFP image not found: {image_path}"
            )

        # Force RGB because HVNet expects:
        #
        # [B, 3, H, W]
        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        # ----------------------------------------------------------
        # Structured clinical indicators
        #
        # These are LongTensor scalars because clinical_encoder.py
        # uses nn.Embedding.
        # ----------------------------------------------------------

        iop = torch.tensor(
            int(
                row[self.iop_column]
            ),
            dtype=torch.long,
        )

        acd = torch.tensor(
            int(
                row[self.acd_column]
            ),
            dtype=torch.long,
        )

        cag = torch.tensor(
            int(
                row[self.cag_column]
            ),
            dtype=torch.long,
        )

        # ----------------------------------------------------------
        # Class label
        # ----------------------------------------------------------

        label = torch.tensor(
            int(
                row[self.label_column]
            ),
            dtype=torch.long,
        )

        # ----------------------------------------------------------
        # Patient ID
        #
        # IMPORTANT:
        # This value is used only for patient-level splitting and
        # traceability. It must never be passed as a predictive
        # model input.
        # ----------------------------------------------------------

        patient_id = str(
            row[self.patient_id_column]
        )

        return {
            "image": image,
            "iop": iop,
            "acd": acd,
            "cag": cag,
            "label": label,
            "patient_id": patient_id,
            "image_path": str(
                image_path
            ),
        }

    # ==============================================================
    # Convenience methods
    # ==============================================================

    def get_patient_ids(
        self,
    ):
        """
        Return all patient IDs.

        Useful for patient-level train/validation/test partitioning.
        """

        return (
            self.metadata[
                self.patient_id_column
            ]
            .astype(str)
            .to_numpy()
        )

    def get_labels(
        self,
    ):
        """
        Return all integer class labels.
        """

        return (
            self.metadata[
                self.label_column
            ]
            .astype(int)
            .to_numpy()
        )

    def get_metadata(
        self,
    ) -> pd.DataFrame:
        """
        Return a copy of the metadata table.
        """

        return self.metadata.copy()


# ======================================================================
# Optional alias
# ======================================================================

GlaucomaDataset = HVNetDataset


# ======================================================================
# Minimal sanity check
# ======================================================================

if __name__ == "__main__":

    """
    Example usage.

    This block does not access any real clinical data.

    Replace the paths below with your local files when testing.
    """

    from torchvision import transforms

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    # --------------------------------------------------------------
    # Example only:
    #
    # dataset = HVNetDataset(
    #     metadata_file="datasets/example_metadata.csv",
    #     root_dir=".",
    #     transform=image_transform,
    # )
    #
    # print("Number of samples:", len(dataset))
    #
    # sample = dataset[0]
    #
    # print("Image shape:", sample["image"].shape)
    # print("IOP:", sample["iop"])
    # print("ACD:", sample["acd"])
    # print("CAG:", sample["cag"])
    # print("Label:", sample["label"])
    # print("Patient ID:", sample["patient_id"])
    # --------------------------------------------------------------

    print(
        "HVNetDataset template loaded successfully."
    )
