from __future__ import annotations

from typing import NamedTuple


class Sample(NamedTuple):
    """A sample of an iris flower. Base class used for all types of samples"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class KnownSample(NamedTuple):
    """A sample of an iris flower with a known spicies."""

    sample: Sample
    species: str


class TrainingKnownSample(NamedTuple):
    """A sample of an iris flower with a known spicies used for training the model."""

    sample: KnownSample


class TestingKnownSample(NamedTuple):
    """A sample of an iris flower with a known spicies used for testing the model."""

    sample: KnownSample


class UnknownSample(NamedTuple):
    """A sample of an iris flower with an unknown spicies."""

    sample: Sample


class ClassifiedKnownSample(NamedTuple):
    """A sample of an iris flower with a known spicies and a classification."""

    sample: KnownSample
    classification: str


class ClassifiedUnknownSample(NamedTuple):
    """A sample of an iris flower with an unknown spicies and a classification."""

    sample: UnknownSample
    classification: str


TrainingList = list[TrainingKnownSample]
TestingList = list[TestingKnownSample]
AnySample = KnownSample | UnknownSample
