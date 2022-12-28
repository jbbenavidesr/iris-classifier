"""Abstract classes for IrisClassifier."""
from collections import defaultdict
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
    pools: defaultdict[bool, list[KnownSample]] = defaultdict(list)
    partition = ((rule(s, i), s) for i, s in enumerate(samples))
    for usage_pool, sample in partition:
        pools[usage_pool].append(sample)

    training_samples = [TrainingKnownSample(s) for s in pools[True]]
    testing_samples = [TestingKnownSample(s) for s in pools[False]]
    return training_samples, testing_samples
