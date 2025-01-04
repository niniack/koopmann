import torch

from koopmann.data import DatasetConfig, get_dataset_class


def test_lotusroot_dataset():
    dataset_config = DatasetConfig(
        dataset_name="LotusRootDataset", num_samples=50, split="train", seed=42
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert dataset.in_features == 2


def test_yin_yang_dataset():
    dataset_config = DatasetConfig(
        dataset_name="YinYangDataset", num_samples=50, split="train", seed=42
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert dataset.in_features == 2
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1, 2]


def test_yin_yang_binary_labels():
    # Non negative label
    dataset_config = DatasetConfig(
        dataset_name="YinYangBinaryDataset",
        num_samples=50,
        split="train",
        seed=42,
        negative_label=False,
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert dataset.in_features == 2
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1]

    # Negative label
    dataset_config = DatasetConfig(
        dataset_name="YinYangBinaryDataset",
        num_samples=50,
        split="train",
        seed=42,
        negative_label=True,
    )
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]


def test_yin_yang_no_dots_binary_label():
    # Non negative label
    dataset_config = DatasetConfig(
        dataset_name="YinYangNoDotsBinaryDataset",
        num_samples=50,
        split="train",
        seed=42,
        negative_label=False,
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert dataset.in_features == 2
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1]

    # Negative label
    dataset_config = DatasetConfig(
        dataset_name="YinYangBinaryDataset",
        num_samples=50,
        split="train",
        seed=42,
        negative_label=True,
    )
    dataset = DatasetClass(config=dataset_config)
    assert len(dataset) == 50
    assert dataset.in_features == 2
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]
