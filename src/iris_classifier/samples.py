from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


@dataclass(frozen=True)
class Sample:
    """A sample of an iris flower. Base class used for all types of samples"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@dataclass(frozen=True)
class KnownSample(Sample):
    """A sample of an iris flower with a known spicies."""

    species: str


@dataclass(frozen=True)
class TrainingKnownSample:
    """A sample of an iris flower with a known spicies used for training the model."""

    sample: KnownSample


@dataclass
class TestingKnownSample:
    """A sample of an iris flower with a known spicies used for testing the model."""

    sample: KnownSample
    classification: str | None = None

    __test__ = False

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.sample.species == self.classification


@dataclass
class UnknownSample:
    """A sample of an iris flower with an unknown spicies."""

    sample: Sample
    classification: str | None = None
