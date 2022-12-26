from __future__ import annotations

import enum
from typing import cast

from .exceptions import InvalidSampleError, OutOfBoundsError


class Species(str, enum.Enum):
    SETOSA = "Iris-setosa"
    VERSICOLOR = "Iris-versicolor"
    VIRGINICA = "Iris-virginica"


class Purpose(enum.IntEnum):
    CLASSIFICATION = 0
    TESTING = 1
    TRAINING = 2


class Sample:
    """A sample of an iris flower."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        """Initialize a sample of an iris flower.

        :param sepal_length: The length of the sepal.
        :param sepal_width: The width of the sepal.
        :param petal_length: The length of the petal.
        :param petal_width: The width of the petal.
        :param species: The spicies of the flower. If the sample is unknown, this value
            should be None.
        """
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __repr__(self) -> str:
        return (
            f"Sample("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width})"
        )

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> Sample:
        """Create a sample from a dictionary.

        :param sample: The sample as a dictionary.
        """
        try:
            sample = cls(
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            )
        except ValueError:
            raise InvalidSampleError(f"Invalid sample in row: {row!r}.")
        except KeyError:
            raise InvalidSampleError(f"Missing column in row: {row!r}")
        else:
            if (
                sample.sepal_length < 0
                or sample.sepal_width < 0
                or sample.petal_length < 0
                or sample.petal_width < 0
            ):
                raise OutOfBoundsError(
                    f"Sample attributes out of bounds in row: {row!r}."
                )
            return sample


class KnownSample(Sample):
    """A sample of an iris flower with a known spicies."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        purpose: Purpose | int,
        species: Species | str,
    ) -> None:
        """Initialize a sample of an iris flower with a known spicies.

        :param sepal_length: The length of the sepal.
        :param sepal_width: The width of the sepal.
        :param petal_length: The length of the petal.
        :param petal_width: The width of the petal.
        :param purpose: The purpose of the sample.
        :param species: The spicies of the flower.
        """
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.TESTING, Purpose.TRAINING}:
            raise ValueError(f"Invalid purpose: {purpose!r}: {purpose_enum}")
        super().__init__(sepal_length, sepal_width, petal_length, petal_width)
        self.species = Species(species)
        self.purpose = purpose_enum
        self._classification: Species | None = None

    @property
    def classification(self) -> Species | None:
        if self.purpose == Purpose.TESTING:
            return self._classification
        else:
            raise AttributeError("classification is only available for testing samples")

    @classification.setter
    def classification(self, value: Species | str) -> None:
        if self.purpose == Purpose.TESTING:
            self._classification = Species(value)
        else:
            raise AttributeError("classification is only available for testing samples")

    def __repr__(self) -> str:
        if self.purpose == Purpose.TESTING:
            name = "TestingKnownSample"

            if self.classification is None:
                classification = ""
            else:
                classification = f", classification={self.classification}"
        else:
            name = "TrainingKnownSample"
            classification = ""

        return (
            f"{name}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species}{classification})"
        )

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.species == self.classification

    @classmethod
    def from_dict(cls, row: dict[str, str], purpose: Purpose | int) -> KnownSample:
        """Create a KnownSample from a dictionary."""

        sample = Sample.from_dict(row)

        try:
            return cls(
                sepal_length=sample.sepal_length,
                sepal_width=sample.sepal_width,
                petal_length=sample.petal_length,
                petal_width=sample.petal_width,
                purpose=purpose,
                species=Species(row["species"]),
            )
        except ValueError:
            raise InvalidSampleError(f"Invalid sample in row: {row!r}")
        except KeyError:
            raise InvalidSampleError(f"Missing column in row: {row!r}")


class UnknownSample(Sample):
    """A sample of an iris flower with an unknown spicies."""

    def __repr__(self) -> str:
        return super().__repr__().replace("Sample", "UnknownSample")

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> UnknownSample:
        return cast(UnknownSample, super().from_dict(row))
