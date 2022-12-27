"""Abstract classes for IrisClassifier."""
from __future__ import annotations

import abc
import random
from typing import Iterable, List, overload

from .samples import SampleDict, TestingKnownSample, TrainingKnownSample


class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.8) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: float = 0.8,
    ) -> None:
        ...

    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: float = 0.8,
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
        training_subset: float = 0.8,
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: int | None = None

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)

    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sample) for sample in self[: self.split]]

    @property
    def testing(self) -> list[TestingKnownSample]:
        self.shuffle()
        return [TestingKnownSample(**sample) for sample in self[self.split :]]


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
