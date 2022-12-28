"""Abstract classes for IrisClassifier."""
from typing import Callable, Iterable

from .samples import KnownSample, TestingKnownSample, TrainingKnownSample


def training_80(s: KnownSample, i: int) -> bool:
    return i % 5 != 0


def training_90(s: KnownSample, i: int) -> bool:
    return i % 10 != 0


def training_75(s: KnownSample, i: int) -> bool:
    return i % 4 != 0


def training_67(s: KnownSample, i: int) -> bool:
    return i % 3 != 0


TrainingList = list[TrainingKnownSample]
TestingList = list[TestingKnownSample]


def partition_samples(
    samples: Iterable[KnownSample],
    rule: Callable[[KnownSample, int], bool],
) -> tuple[TrainingList, TestingList]:
    """Partition samples into training and testing sets."""
    training_samples = list(
        TrainingKnownSample(sample)
        for i, sample in enumerate(samples)
        if rule(sample, i)
    )
    testing_samples = list(
        TestingKnownSample(sample)
        for i, sample in enumerate(samples)
        if not rule(sample, i)
    )
    return training_samples, testing_samples
