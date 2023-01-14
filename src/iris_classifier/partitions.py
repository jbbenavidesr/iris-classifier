"""Abstract classes for IrisClassifier."""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, Iterator, List

from .models import (
    KnownSample,
    TestingKnownSample,
    TestingList,
    TrainingKnownSample,
    TrainingList,
)


def training_80(i: int) -> bool:
    return i % 5 != 0


def training_90(i: int) -> bool:
    return i % 10 != 0


def training_75(i: int) -> bool:
    return i % 4 != 0


def training_67(i: int) -> bool:
    return i % 3 != 0


ModuloDict = DefaultDict[int, List[KnownSample]]


def partition_samples(
    samples: Iterable[KnownSample], training_rule: Callable[[int], bool]
) -> tuple[TrainingList, TestingList]:
    """Partitions the samples into training and testing sets."""
    rule_multiple = 60
    partitions: ModuloDict = defaultdict(list)
    for s in samples:
        partitions[hash(s) % rule_multiple].append(s)

    training_partitions: list[Iterator[TrainingKnownSample]] = []
    testing_partitions: list[Iterator[TestingKnownSample]] = []
    for i, p in enumerate(partitions.values()):
        if training_rule(i):
            training_partitions.append((TrainingKnownSample(s) for s in p))
        else:
            testing_partitions.append((TestingKnownSample(s) for s in p))

    training = list(itertools.chain(*training_partitions))
    testing = list(itertools.chain(*testing_partitions))

    return training, testing
