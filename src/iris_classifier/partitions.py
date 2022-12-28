"""Abstract classes for IrisClassifier."""
from __future__ import annotations

import abc
import random
from typing import Iterable, List, overload

from .samples import SampleDict, TestingKnownSample, TrainingKnownSample, KnownSample


class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: tuple[int, int] = (8, 10)) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        ...

    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        self.training_subset = training_subset
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()

    @property
    @abc.abstractmethod
    def training(self) -> list[TrainingKnownSample]:
        ...

    @property
    @abc.abstractmethod
    def testing(self) -> list[TestingKnownSample]:
        ...


class ShufflingSamplePartition(SamplePartition):
    """Implementation of partition that shuffles the list and cuts it."""

    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: int | None = None

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            n, d = self.training_subset
            self.split = int(len(self) * n / d)

    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return [
            TrainingKnownSample(KnownSample(**sample)) for sample in self[: self.split]
        ]

    @property
    def testing(self) -> list[TestingKnownSample]:
        self.shuffle()
        return [
            TestingKnownSample(KnownSample(**sample)) for sample in self[self.split :]
        ]


class DealingPartition(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        items: Iterable[SampleDict] | None,
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        ...

    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...

    @abc.abstractmethod
    def append(self, item: SampleDict) -> None:
        ...

    @property
    @abc.abstractmethod
    def training(self) -> list[TrainingKnownSample]:
        ...

    @property
    @abc.abstractmethod
    def testing(self) -> list[TestingKnownSample]:
        ...


class CountingDealingPartition(DealingPartition):
    def __init__(
        self,
        items: Iterable[SampleDict] | None = None,
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items is not None:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(KnownSample(**item)))
        else:
            self._testing.append(TestingKnownSample(KnownSample(**item)))
        self.counter += 1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training

    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing
