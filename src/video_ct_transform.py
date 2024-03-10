""" Video transforms for CT data """

from typing import Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import Randomizable
from monai.transforms.transform import MapTransform


class SampleVideoFromVolumeD(Randomizable, MapTransform):
    """
    Sample a fixed number of frames from a volume.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_frames: int = 16,
        random: bool = False,
    ) -> None:
        super(Randomizable, self).__init__(keys)
        self.random = random
        self.num_frames = num_frames

    def sample_frames(self, x: np.ndarray) -> np.ndarray:
        """
        Sample a fixed number of frames from a volume.

        Args:
            x (np.ndarray): Input volume

        Returns:
            np.ndarray: Sampled frames
        """
        depth = x.shape[1]

        # Generate frames + 2 equidistant points in the range [0, depth)
        indices = np.round(np.linspace(0, depth - 1, self.num_frames + 2)).astype(int)

        # remove two points at the beginning and end
        indices = indices[1:-1]

        if self.random:
            # randomly move indices by half of distance of slices
            slice_distance = indices[1] - indices[0]
            shift = np.random.randint(-slice_distance // 2, slice_distance // 2)
            indices = indices + shift

        # Use these indices to slice the volume
        return x[:, indices, :, :]

    def convert_rgb(self, x: np.ndarray) -> np.ndarray:
        """
        Convert grayscale to RGB by repeating the same frame three times.

        Args:
            x (np.ndarray): Input volume

        Returns:
            np.ndarray: RGB volume
        """
        return np.concatenate((x, x, x), axis=0)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Mapping[Hashable, np.ndarray]:
        """
        Sample a fixed number of frames from a volume.

        Args:
            data (Mapping[Hashable, np.ndarray]): Input data

        Returns:
            Mapping[Hashable, np.ndarray]: Transformed data
        """
        d = dict(data)
        for key in self.keys:
            d[key] = self.sample_frames(d[key])
            d[key] = self.convert_rgb(d[key])
        return d


class SampleVideoEnsembleFromVolumeD(MapTransform):
    """
    Sample a fixed number of frames from a volume and create an ensemble.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_frames: int = 16,
    ) -> None:
        super().__init__(keys)
        self.num_frames = num_frames

    def sample_frames(self, x: np.ndarray) -> np.ndarray:
        """
        Sample a fixed number of frames from a volume and create an ensemble.

        Args:
            x (np.ndarray): Input volume

        Returns:
            np.ndarray: Sampled frames
        """
        depth = x.shape[1]

        # Generate frames + 2 equidistant points in the range [0, depth)
        indices = np.round(np.linspace(0, depth - 1, self.num_frames + 2)).astype(int)

        # remove two points at the beginning and end
        indices = indices[1:-1]

        slice_distance = indices[1] - indices[0]
        ensemble = np.stack(
            [
                torch.cat(
                    (
                        x[:, indices + i, :, :],
                        x[:, indices + i, :, :],
                        x[:, indices + i, :, :],
                    ),
                    dim=0,
                )
                for i in range(-slice_distance // 2, slice_distance // 2)
            ]
        )

        # Use these indices to slice the volume
        return ensemble

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Mapping[Hashable, np.ndarray]:
        """
        Sample a fixed number of frames from a volume and create an ensemble.

        Args:
            data (Mapping[Hashable, np.ndarray]): Input data

        Returns:
            Mapping[Hashable, np.ndarray]: Transformed data
        """
        d = dict(data)
        for key in self.keys:
            d[key] = self.sample_frames(d[key])
        return d
