__all__ = [
    "YinYangDataset",
    "YinYangBinaryDataset",
    "MNISTDataset",
]

import sys
from typing import Callable, Literal

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    dataset_name: Literal[
        "YinYangDataset",
        "YinYangNoDotsBinaryDataset",
        "YinYangBinaryDataset",
        "LotusRootDataset",
        "SunflowerDataset",
        "TorusDataset",
        "MNISTDataset",
        "CIFAR10Dataset",
        "FashionMNISTDataset",
    ]
    num_samples: int
    split: str
    torch_transform: Callable | None = None
    seed: int | None = 42
    negative_label: bool = False


def get_dataset_class(name: str) -> Dataset:
    if name == "SunflowerDataset":
        name = "LotusRootDataset"
    return getattr(sys.modules[__name__], name)


def create_data_loader(
    dataset: Dataset, batch_size: Int, global_seed: Int = 0, shuffle: bool = True
) -> DataLoader:
    loader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


class BaseYinYangDataset(Dataset):
    """
    A YinYang inspired dataset, adapted from: \\
    https://github.com/lkriener/yin_yang_data_set
    """

    def __init__(
        self,
        config: DatasetConfig,
        binary: Bool = False,
        dots: Bool = True,  # Overriden to True if binary is True
        r_small: Float = 0.1,
        r_big: Float = 0.5,
        rotate: Bool = False,
    ):
        super().__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(config.seed)
        self.transform = config.torch_transform
        self.r_small = r_small
        self.r_big = r_big

        if binary:
            self.class_names = ["yin", "yang"]
        else:
            self.class_names = ["yin", "yang", "dot"]
        self.features = []
        self.labels = []
        self.in_features = 2

        self.binary = binary
        if not binary and not dots:
            self.dots = True
        else:
            self.dots = dots

        for i in range(config.num_samples):
            goal_class = self.rng.randint(2 if binary else 3)
            x, y, c = self.get_sample(goal=goal_class)

            if c == 0 and (binary and config.negative_label):
                c = -1

            val = np.array([x, y])
            self.features.append(val)
            self.labels.append(c)

        self.features = torch.FloatTensor(np.asarray(self.features))
        if rotate:
            rot_matr = torch.Tensor([[0, -1], [1, 0]])
            self.features = self.features @ rot_matr
        self.labels = torch.IntTensor(self.labels).unsqueeze(1)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2.0 * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles and (not self.binary or (self.binary and not self.dots)):
            return 2
        elif is_circles and self.binary:
            return int(criterion1 or criterion2)

        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    @property
    def name(self):
        raise NotImplementedError("Subclasses must implement the 'name' property")

    def __getitem__(self, index):
        sample = (self.features[index], self.labels[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.labels)

    def __str__(self) -> str:
        return self.name


class YinYangDataset(BaseYinYangDataset):
    """
    3-way YinYang classification (yin, yang, dots)
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=False,
            dots=True,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangDataset"


class YinYangNoDotsBinaryDataset(BaseYinYangDataset):
    """
    2-way YinYang classification (yin, yang). Dots have been removed.
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=True,
            dots=False,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangNoDotsBinaryDataset"


class YinYangBinaryDataset(BaseYinYangDataset):
    """
    2-way (canonical) YinYang classification (yin, yang). Dots are part of Yin/Ying class.
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=True,
            dots=True,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangBinaryDataset"


class LotusRootDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        r_inner=0.35,
        r_outer=1,
        num_small_circles=9,
    ):
        super().__init__()
        self.num_samples = config.num_samples
        self.r_inner = r_inner  # Radius of the central circle
        self.r_outer = r_outer  # Radius of the overall circle
        self.num_small_circles = num_small_circles  # Number of small circles around the center
        self.rng = np.random.RandomState(config.seed)
        self.features = []
        self.labels = []

        for _ in range(config.num_samples):
            x, y, label = self.sample_point()
            self.features.append([x, y])
            self.labels.append(label)

        self.features = torch.FloatTensor(np.asarray(self.features))
        self.labels = torch.IntTensor(self.labels).unsqueeze(1)
        self.in_features = 2

    def sample_point(self):
        found_sample = False
        while not found_sample:
            # Generate random points within the bounding circle
            x, y = self.rng.uniform(-self.r_outer, self.r_outer, 2)
            if np.sqrt(x**2 + y**2) > self.r_outer:
                continue

            label = self.classify_point(x, y)
            if label is not None:
                found_sample = True

        return x, y, label

    def classify_point(self, x, y):
        # Check if point is inside the central circle
        distance = np.sqrt(x**2 + y**2)
        if distance < self.r_inner / 2:
            return 1  # Class 0: Central Circle

        # Check if point is inside any of the smaller surrounding circles
        angle_step = 2 * np.pi / (self.num_small_circles - 1)
        for i in range(self.num_small_circles - 1):
            angle = i * angle_step
            cx = (2 / 3) * self.r_outer * np.cos(angle)  # Center x of the small circle
            cy = (2 / 3) * self.r_outer * np.sin(angle)  # Center y of the small circle
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < self.r_inner / 2:
                return 1  # Class 1: Surrounding Small Circle

        return 0  # Reject point if it doesn't fit any class

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def name(self):
        return "LotusRootDataset"


class TorusDataset(Dataset):
    def __init__(self, config, R=1.0, r=0.3, num_pairs=2):
        super().__init__()
        self.num_samples = config.num_samples
        self.R = R  # Major radius (distance from the center of the tube to the center of the torus)
        self.r = r  # Minor radius (radius of the tube)
        self.num_pairs = num_pairs  # Number of interlocking ring pairs
        self.rng = np.random.RandomState(config.seed)
        self.features = []
        self.labels = []
        self.in_features = 3

        # Split samples evenly among all toruses
        num_samples_per_torus = self.num_samples // (2 * self.num_pairs)

        for pair in range(self.num_pairs):
            # Offset for the current pair in space to keep toruses distinct and interlocked
            offset_x = 4 * self.R * pair  # Offset in the x-direction
            interlock_offset = self.R  # Offset in the z-direction for interlocking

            # Generate points for the first torus in the XY-plane
            for _ in range(num_samples_per_torus):
                x, y, z = self.sample_point_xy_plane()
                self.features.append([x + offset_x, y, z])  # Apply the x-offset
                self.labels.append(0)  # Label for the first torus in this pair

            # Generate points for the second torus in the XZ-plane, with a z-offset for interlocking
            for _ in range(num_samples_per_torus):
                x, y, z = self.sample_point_xz_plane()
                self.features.append(
                    [x + offset_x + interlock_offset, y, z]
                )  # Apply x and z offsets
                self.labels.append(1)  # Label for the second torus in this pair

        self.features = torch.FloatTensor(np.asarray(self.features))
        self.labels = torch.IntTensor(self.labels).unsqueeze(1)

    def sample_point_xy_plane(self):
        # Generate a point on a torus lying in the XY-plane
        theta = self.rng.uniform(0, 2 * np.pi)
        phi = self.rng.uniform(0, 2 * np.pi)

        x = (self.R + self.r * np.cos(phi)) * np.cos(theta)
        y = (self.R + self.r * np.cos(phi)) * np.sin(theta)
        z = self.r * np.sin(phi)

        return x, y, z

    def sample_point_xz_plane(self):
        # Generate a point on a torus lying in the XZ-plane
        theta = self.rng.uniform(0, 2 * np.pi)
        phi = self.rng.uniform(0, 2 * np.pi)

        x = (self.R + self.r * np.cos(phi)) * np.cos(theta)
        y = self.r * np.sin(phi)  # y-coordinate varies based on minor radius
        z = (self.R + self.r * np.cos(phi)) * np.sin(theta)

        return x, y, z

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def name(self):
        return "TorusDataset"


class MNISTDataset(datasets.MNIST):
    """Simple wrapper around the MNIST dataset with default configurations."""

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Scale to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize inputs
        ]
    )

    def __init__(
        self,
        config=None,
        seed=42,
        transform=None,  # Torch transforms, uses default MNIST transform if None
        root="/scratch/nsa325/datasets/",  # Dataset location
    ):
        self.transform = transform or self.default_transform
        train = True if config.split == "train" else False
        super().__init__(root=root, train=train, download=True, transform=self.transform)
        self.seed = seed
        self.config = config
        self.in_features = 784
        self.labels = self.targets

    def name(self):
        return "MNISTDataset"


class FashionMNISTDataset(datasets.FashionMNIST):
    """Simple wrapper around the FashionMNIST dataset with default configurations."""

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Scale to [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize inputs to [-1, 1] range
        ]
    )

    def __init__(
        self,
        config=None,
        seed=42,
        transform=None,  # Torch transforms, uses default FashionMNIST transform if None
        root="/scratch/nsa325/datasets/",  # Dataset location
    ):
        self.transform = transform or self.default_transform
        train = True if config.split == "train" else False
        super().__init__(root=root, train=train, download=True, transform=self.transform)
        self.seed = seed
        self.config = config
        self.in_features = 784  # FashionMNIST images are 28x28
        self.labels = self.targets

    def name(self):
        return "FashionMNISTDataset"


class CIFAR10Dataset(datasets.CIFAR10):
    """Simple wrapper around the CIFAR-10 dataset with default configurations."""

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to tensors and scale to [0, 1]
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),  # Mean for CIFAR-10
                std=(0.2470, 0.2435, 0.2616),  # Standard deviation for CIFAR-10
            ),  # Normalize inputs
        ]
    )

    def __init__(
        self,
        config=None,
        seed=42,
        transform=None,  # Torch transforms, uses default CIFAR-10 transform if None
        root="/scratch/nsa325/datasets/",  # Dataset location
    ):
        self.transform = transform or self.default_transform
        train = True if config.split == "train" else False
        super().__init__(root=root, train=train, download=True, transform=self.transform)
        self.seed = seed
        self.config = config
        self.in_features = 3072  # CIFAR-10 has 32x32x3 inputs
        self.labels = self.targets

    def name(self):
        return "CIFAR10Dataset"
