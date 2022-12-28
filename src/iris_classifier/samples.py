from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


@dataclass
class Sample:
    """A sample of an iris flower. Base class used for all types of samples"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@dataclass
class KnownSample(Sample):
    """A sample of an iris flower with a known spicies."""

    species: str


@dataclass
class TrainingKnownSample(KnownSample):
    """A sample of an iris flower with a known spicies used for training the model."""

    pass


@dataclass
class TestingKnownSample(KnownSample):
    """A sample of an iris flower with a known spicies used for testing the model."""

    __test__ = False

    classification: str | None = None

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.species == self.classification


@dataclass
class UnknownSample(Sample):
    """A sample of an iris flower with an unknown spicies."""

    pass


@dataclass
class ClassifiedSample(Sample):
    """Created from the sample provided by the user and the result of classification"""

    classification: str
