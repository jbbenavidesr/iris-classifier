"""Abstract classes for IrisClassifier."""
from __future__ import annotations

from typing import List, overload, Iterable
import abc

from .samples import KnownSample, SampleDict


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
    def training(self) -> list[KnownSample]:
        ...

    @property
    @abc.abstractmethod
    def testing(self) -> list[KnownSample]:
        ...
