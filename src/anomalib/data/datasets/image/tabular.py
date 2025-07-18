# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Tabular Dataset.

This module provides a custom PyTorch Dataset implementation for loading
images using a selection of paths and labels defined in a table or tabular file.
It does not require a specific folder structure and allows subsampling and
relabeling without moving files. The dataset supports both classification and
segmentation tasks.

The table should contain columns for ``image_paths``, ``label_index``, ``split``,
and optionally ``masks_paths`` for segmentation tasks.

Example:
    >>> from anomalib.data.datasets import TabularDataset
    >>> samples = {
    ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
    ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
    ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
    ... }
    >>> dataset = TabularDataset(
    ...     name="custom",
    ...     samples=samples,
    ...     root="./datasets/custom",
    ... )
"""

from pathlib import Path

import numpy as np
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split


class TabularDataset(AnomalibDataset):
    """Dataset class for loading images from paths and labels defined in a table.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible ``list``
            or ``dict`` containing the dataset information.
        augmentations (Transform | None, optional): Augmentations to apply to the images.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Examples:
        Create a classification dataset:

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> from anomalib.data.datasets import TabularDataset
        >>> transform = get_transforms(
        ...     image_size=256,
        ...     normalization=InputNormalizationMethod.NONE
        ... )
        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )

        Create a segmentation dataset:

        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ...     "mask_path": ["masks/mask1.png", "masks/mask2.png", "masks/mask3.png", ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        name: str,
        samples: dict | list | DataFrame,
        augmentations: Transform | None = None,
        root: str | Path | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.samples = make_tabular_dataset(
            samples=samples,
            root=self.root,
            split=self.split,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name


def make_tabular_dataset(
    samples: dict | list | DataFrame,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> DataFrame:
    """Create a dataset from a table of image paths and labels.

    Args:
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible
            ``list`` or ``dict`` containing the dataset information.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for image paths, labels, splits
            and mask paths (for segmentation).

    Examples:
        Create a classification dataset:
        >>> samples = {
        ...     "image_path": ["images/00.png", "images/01.png", "images/02.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.NORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TRAIN, ... ],
        ... }
        >>> tabular_df = make_tabular_dataset(
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     split=Split.TRAIN,
        ... )
        >>> tabular_df.head()
           image_path                         label            label_index    mask_path    split
        0  ./datasets/custom/images/00.png    DirType.NORMAL    0                           Split.TRAIN
        1  ./datasets/custom/images/01.png    DirType.NORMAL    0                           Split.TRAIN
        2  ./datasets/custom/images/02.png    DirType.NORMAL    0                           Split.TRAIN
        3  ./datasets/custom/images/03.png    DirType.NORMAL    0                           Split.TRAIN
        4  ./datasets/custom/images/04.png    DirType.NORMAL    0                           Split.TRAIN
    """
    # Convert to pandas DataFrame if dictionary or list is given
    if isinstance(samples, (dict, list)):
        samples = DataFrame(samples)
    if "image_path" not in samples.columns:
        msg = "The samples table must contain an 'image_path' column."
        raise ValueError(msg)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Adding missing columns
    existing_cols = samples.columns
    has_label_index = "label_index" in existing_cols
    has_label = "label" in existing_cols
    has_split = "split" in existing_cols

    if has_label_index:
        samples["label_index"] = samples["label_index"].astype("Int64")

    if not (has_label_index or has_label or has_split):
        msg = "The samples table must contain at least one of 'label_index', 'label' or 'split' columns."
        raise ValueError(msg)

    # Add missing columns with vectorized mapping
    # label_index missing, label present
    if not has_label_index and has_label:
        label_dict = {
            DirType.ABNORMAL: LabelName.ABNORMAL,
            DirType.NORMAL: LabelName.NORMAL,
            DirType.NORMAL_TEST: LabelName.NORMAL,
        }
        samples["label_index"] = samples["label"].map(label_dict).astype("Int64")
        has_label_index = True

    # label_index and label missing, split present
    if not has_label_index and not has_label and has_split:
        split_dict = {
            Split.TRAIN: LabelName.NORMAL,
            Split.TEST: LabelName.ABNORMAL,
        }
        samples["label_index"] = samples["split"].map(split_dict).astype("Int64")
        has_label_index = True

    # label and split missing, label_index present
    if has_label_index and not has_label and not has_split:
        index_dict = {
            LabelName.ABNORMAL: DirType.ABNORMAL,
            LabelName.NORMAL: DirType.NORMAL,
        }
        samples["label"] = samples["label_index"].map(index_dict)
        has_label = True

    # Recalculate for possible previous additions
    existing_cols = samples.columns
    has_label_index = "label_index" in existing_cols
    has_label = "label" in existing_cols
    has_split = "split" in existing_cols

    # label missing, label_index and split present
    if has_label_index and not has_label and has_split:
        normal_mask = samples["label_index"] == LabelName.NORMAL
        train_mask = samples["split"] == Split.TRAIN
        test_mask = samples["split"] == Split.TEST
        label_arr = np.full(samples.shape[0], None, dtype=object)
        label_arr[normal_mask & train_mask] = DirType.NORMAL
        label_arr[normal_mask & test_mask] = DirType.NORMAL_TEST
        label_arr[(samples["label_index"] == LabelName.ABNORMAL)] = DirType.ABNORMAL
        samples["label"] = label_arr
        has_label = True

    # split missing, label_index and label present
    if has_label_index and has_label and not has_split:
        split_dict = {
            DirType.NORMAL: Split.TRAIN,
            DirType.ABNORMAL: Split.TEST,
            DirType.NORMAL_TEST: Split.TEST,
        }
        samples["split"] = samples["label"].map(split_dict)
        has_split = True

    # Add mask_path column if not exists
    if "mask_path" not in samples.columns:
        samples["mask_path"] = ""

    # Fillna in mask_path once
    samples["mask_path"] = samples["mask_path"].fillna("")

    # Add root to paths using vectorized string ops for efficiency
    if root:
        root = str(root)
        if not root.endswith(("/", "\\")):
            root = root + "/"
        # Fastest path join for str columns using .str.cat or .astype(str).radd()
        samples["image_path"] = np.where(
            samples["image_path"].astype(str).str.startswith(root),
            samples["image_path"].astype(str),
            root + samples["image_path"].astype(str),
        )
        mask_mask = samples["mask_path"] != ""
        if mask_mask.any():
            samples.loc[mask_mask, "mask_path"] = np.where(
                samples.loc[mask_mask, "mask_path"].astype(str).str.startswith(root),
                samples.loc[mask_mask, "mask_path"].astype(str),
                root + samples.loc[mask_mask, "mask_path"].astype(str),
            )

    # Convert columns to correct dtype (faster to just do minimal types)
    samples["image_path"] = samples["image_path"].astype(str)
    samples["mask_path"] = samples["mask_path"].astype(str)
    if has_label:
        samples["label"] = samples["label"].astype(str)

    # Check if anomalous samples are in training set
    # Use numpy vectorized boolean instead of slow pandas chained logic
    anomaly_in_train = ((samples["label_index"] == LabelName.ABNORMAL) & (samples["split"] == Split.TRAIN)).any()
    if anomaly_in_train:
        msg = "Training set must not contain anomalous samples."
        raise MisMatchError(msg)

    # Check for None or NaN values
    if samples.isna().values.any():
        msg = "The samples table contains None or NaN values."
        raise ValueError(msg)

    # Infer the task type
    if (samples["mask_path"] == "").all():
        samples.attrs["task"] = "classification"
    else:
        samples.attrs["task"] = "segmentation"

    # Filter by split if required
    if split is not None:
        samples = samples[samples["split"] == split].reset_index(drop=True)

    return samples
