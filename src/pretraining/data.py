""" Data module for pretraining """

from pathlib import Path
from typing import Dict, Hashable, Mapping

import pandas as pd
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import Dataset
from monai.transforms import (
    Compose,
    CropForegroundD,
    EnsureChannelFirstD,
    LoadImageD,
    RandFlipD,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
)
from monai.transforms.transform import MapTransform

from src.video_ct_transform import SampleVideoFromVolumeD


def get_dataset(dataset_source_path: str, csv_file_name: str, args: dict) -> Dataset:
    """
    Get the training dataset.

    Args:
        dataset_source_path (str): Path to the dataset
        csv_file_name (str): Name of the csv file
        args (dict): Arguments

    Returns:
        Dataset: Training dataset
    """
    data_dir = Path(dataset_source_path)

    # load csv file
    csv_path = data_dir / csv_file_name
    df = pd.read_csv(csv_path, index_col=0)

    # fix to check for non-existing files
    if "image" in df.columns:
        df = df[df["image"].apply(lambda p: (data_dir / p).is_file())]

    # get volume paths
    volume_paths = list(df["image"])
    levels = list(df["level_idx"])
    train_files = [
        {
            "image": str(data_dir / path),
            "mask": str(data_dir / path.replace("/raw/", "/seg/")),
            "level_idx": level,
        }
        for path, level in zip(volume_paths, levels)
    ]
    dataset_train = Dataset(data=train_files, transform=get_transform(args))
    print(f"Dataset all training: number of data: {len(train_files)}")
    return dataset_train


def get_transform(args: dict) -> Compose:
    """
    Get the transformation for the dataset.

    Args:
        args (dict): Arguments

    Returns:
        Compose: Transformation for the dataset
    """
    return Compose(
        [
            LoadImageD(keys=["image", "mask"]),
            EnsureChannelFirstD(keys=["image", "mask"]),
            LabelmapToMaskD(keys="mask"),
            ScaleIntensityRangeD(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
            CropForegroundD(
                keys=["image"],
                source_key="mask",
                allow_smaller=True,
                margin=(0, 16, 100),
            ),
            ResizeWithPadOrCropD(
                keys=["image"],
                mode=["constant"],
                spatial_size=(-1, args.input_size, args.input_size),
            ),
            SampleVideoFromVolumeD(keys=["image"], num_frames=16, random=True),
            ResizeWithPadOrCropD(
                keys=["image"],
                mode=["constant"],
                spatial_size=(16, args.input_size, args.input_size),
            ),
            RandFlipD(keys=["image"], spatial_axis=1, prob=0.5),
        ]
    )


class LabelmapToMaskD(MapTransform):
    """
    Convert labelmap to mask.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_label_key: str = "level_idx",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys (KeysCollection): Keys of the corresponding items to be transformed
            mask_label_key (str, optional): Key of the label to be used for masking
            allow_missing_keys (bool, optional): Do not raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.mask_label_key = mask_label_key

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Convert labelmap to a single mask.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): Input data

        Returns:
            Dict[Hashable, NdarrayOrTensor]: Transformed data
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] == d[self.mask_label_key]
        return d
