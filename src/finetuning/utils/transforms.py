""" This module contains the transforms used in the data pipeline """

from pathlib import Path
from typing import Dict, Hashable, Mapping

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    CenterSpatialCropD,
    CropForegroundD,
    EnsureChannelFirstD,
    LambdaD,
    LoadImageD,
    OrientationD,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
)
from monai.transforms.transform import MapTransform
from omegaconf import DictConfig

from src.video_ct_transform import (
    SampleVideoEnsembleFromVolumeD,
    SampleVideoFromVolumeD,
)


def get_all_transforms(cfg: DictConfig, data_dir: Path) -> dict:
    """
    Get all possible transforms for the data pipeline.

    Args:
        cfg (DictConfig): Configuration file
        data_dir (Path): Path to the data directory

    Returns:
        dict: Dictionary containing all possible transforms
    """
    transform_groups = {}

    # define keys used for transforms

    # load image
    transform_groups["load_image"] = get_image_loading_transform(data_dir, cfg)

    # normalize image
    transform_groups["normalize_image"] = get_normalization_transform(cfg)

    # load mask
    transform_groups["load_mask"] = get_mask_loading_transform(data_dir, cfg)

    # crop image
    transform_groups["crop"] = get_crop_transform(cfg)

    # video conversion
    transform_groups["sample_frames"] = get_video_conversion_transform(cfg)

    # video ensemble conversion
    transform_groups[
        "sample_frames_ensemble"
    ] = get_video_ensemble_conversion_transform(cfg)

    return transform_groups


def get_image_loading_transform(data_dir: Path, cfg: DictConfig) -> list:
    """
    Get the transforms for loading the image.

    Args:
        data_dir (Path): Path to the data directory
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    transforms = [
        LambdaD(keys="image", func=lambda p: data_dir / "raw" / p),
        LoadImageD(keys="image"),
        EnsureChannelFirstD(keys="image"),
    ]

    # add orientation adjustment if specified in configuration
    if cfg.data.orientation:
        transforms.append(OrientationD(keys="image", axcodes=cfg.data.orientation))

    return transforms


def get_normalization_transform(cfg: DictConfig) -> list:
    """
    Get the transforms for normalizing the image.

    Args:
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    return [
        ScaleIntensityRangeD(
            keys="image",
            a_min=cfg.data.a_min,
            a_max=cfg.data.a_max,
            b_min=cfg.data.b_min,
            b_max=cfg.data.b_max,
            clip=True,
        ),
    ]


def get_mask_loading_transform(data_dir: Path, cfg: DictConfig) -> list:
    """
    Get the transforms for loading the mask.

    Args:
        data_dir (Path): Path to the data directory
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    transforms = [
        LambdaD(keys="mask", func=lambda p: data_dir / "seg" / p),
        LoadImageD(keys="mask"),
        EnsureChannelFirstD(keys="mask"),
        LabelmapToMaskD(keys="mask"),
    ]

    # add orientation adjustment if specified in configuration
    if cfg.data.orientation:
        transforms.append(OrientationD(keys="mask", axcodes=cfg.data.orientation))

    return transforms


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


def get_crop_transform(cfg: DictConfig) -> list:
    """
    Get the transforms for cropping the image.

    Args:
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    # apply crop based on segmentation mask if 'load_mask' is in the configuration
    if "load_mask" in cfg.data.transforms:
        adjusted_input_size = cfg.data.input_size.copy()
        adjusted_input_size[0] = -1
        return [
            CropForegroundD(
                keys="image",
                source_key="mask",
                allow_smaller=True,
                margin=(0, 16, 100),
            ),
            ResizeWithPadOrCropD(
                keys="image",
                mode="constant",
                spatial_size=adjusted_input_size,
            ),
        ]
    else:
        # apply central spatial crop if 'load_mask' is not in the configuration
        return [CenterSpatialCropD(keys="image", roi_size=cfg.data.input_size)]


def get_video_conversion_transform(cfg: DictConfig) -> list:
    """
    Get the transforms for converting the volume to a video.

    Args:
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    return [
        SampleVideoFromVolumeD(
            keys=["image"], num_frames=16, random=cfg.data.random_sampling
        ),
        ResizeWithPadOrCropD(
            keys=["image"],
            mode=["constant"],
            spatial_size=(16, cfg.model.input_size, cfg.model.input_size),
        ),
    ]


def get_video_ensemble_conversion_transform(cfg: DictConfig) -> list:
    """
    Get the transforms for converting the volume to a video ensemble.

    Args:
        cfg (DictConfig): Configuration file

    Returns:
        list: List of transforms
    """
    return [SampleVideoEnsembleFromVolumeD(keys=["image"], num_frames=16)]
