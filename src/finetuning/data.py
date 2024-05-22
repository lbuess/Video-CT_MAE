""" Data loading and transformation functions """

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from omegaconf import DictConfig

from src.finetuning.utils.transforms import get_all_transforms


def get_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for training, validation, and test set.

    Args:
        cfg (DictConfig): Configuration file

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test data loader
    """
    data_dir = Path(cfg.data.source_path)
    logging.info(f"Loading data from {data_dir}")

    # load csv file (stores information for each vertebra volume)
    df = pd.read_csv(data_dir / cfg.data.csv_file_name, index_col=0)

    # filter data frame
    df = df.loc[
        df["image"].apply(lambda p: (data_dir / "raw" / str(p)).is_file())
    ].query(  # remove non-existing files
        "@cfg.data.min_vertebra_level <= level_idx <= @cfg.data.max_vertebra_level and fx != -1"
    )  # filter by vertebra level

    # duplicate image paths to create a mask path
    df["mask"] = df.image

    # get training, validation and test datasets
    splits = ["training", "validation", "test"]
    datasets = prepare_datasets(df, splits, cfg, data_dir)

    # create data loaders
    loaders = {
        "training": create_data_loader(
            datasets["training"], cfg, shuffle=True, batch_size=cfg.batch_size
        ),
        "validation": create_data_loader(
            datasets["validation"],
            cfg,
            shuffle=False,
            batch_size=(cfg.batch_size if not cfg.ensemble_strategy else 1),
        ),
        "test": create_data_loader(datasets["test"], cfg, shuffle=False, batch_size=1),
    }

    # Logging dataset sizes
    for split, loader in loaders.items():
        logging.info(f"{split.capitalize()} split: {len(loader.dataset)} samples")

    # Returning the data loaders
    train_loader, valid_loader, test_loader = (
        loaders["training"],
        loaders["validation"],
        loaders["test"],
    )
    return train_loader, valid_loader, test_loader


def create_data_loader(
    dataset: Dataset, cfg: DictConfig, shuffle: bool, batch_size: int
) -> DataLoader:
    """
    Utility function to create a DataLoader.

    Args:
        dataset (Dataset): Dataset to be loaded
        cfg (DictConfig): Configuration file
        shuffle (bool): Shuffle the data
        batch_size (int): batch size

    Returns:
        DataLoader: Data loader for the given dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=shuffle,  # Only drop last in training
    )


def prepare_datasets(
    df: pd.DataFrame, splits: list, cfg: DictConfig, data_dir: Path
) -> Dict[str, Dataset]:
    """
    Prepare datasets for each split.

    Args:
        df (pd.DataFrame): Dataframe containing the data
        splits (list): Data splits
        cfg (DictConfig): Configuration file
        data_dir (Path): Directory containing the data

    Returns:
        dict[str, Dataset]: Dataset for each split
    """
    datasets = {
        split: Dataset(
            df[df["split"] == split].to_dict("records"),
            transform=get_transforms(data_dir, cfg, split),
        )
        for split in splits
    }
    return datasets


def get_transforms(data_dir: Path, cfg: DictConfig, split: str) -> Compose:
    """
    Get data transformations based on the config file.

    Args:
        data_dir (Path): Directory containing the data
        cfg (DictConfig): Configuration file
        split (str): Data splits

    Returns:
        Compose: Series of transformations
    """
    # default transformations
    transforms_collection = ["load_image", "normalize_image"]

    # load mask transform
    if "load_mask" in cfg.data.transforms:
        transforms_collection.insert(0, "load_mask")

    # crop transform
    if "crop" in cfg.data.transforms:
        transforms_collection.append("crop")

    # sample frames transform (convert to video format)
    if "sample_frames" in cfg.data.transforms:
        transform = (
            "sample_frames_ensemble"
            if cfg.ensemble_strategy and split in ["validation", "test"]
            else "sample_frames"
        )
        transforms_collection.append(transform)

    # get all possible transforms
    transforms_options = get_all_transforms(cfg, data_dir)

    # combine all transforms
    transforms = []
    for k in transforms_collection:
        transforms.extend(transforms_options[k])

    return Compose(transforms)
