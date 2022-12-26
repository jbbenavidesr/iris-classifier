from __future__ import annotations

from typing import Optional

from .exceptions import InvalidSampleError


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
        self.classification: Optional[str] = None

    def __repr__(self) -> str:
        if self.classification is None:
            classification = ""
        else:
            classification = f", classification={self.classification}"

        return (
            f"Sample("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}{classification})"
        )

    def classify(self, classification: str) -> None:
        """Classify the sample.

        :param classification: The classification of the sample.
        """
        self.classification = classification


class KnownSample(Sample):
    """A sample of an iris flower with a known spicies."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: str,
    ) -> None:
        """Initialize a sample of an iris flower with a known spicies.

        :param sepal_length: The length of the sepal.
        :param sepal_width: The width of the sepal.
        :param petal_length: The length of the petal.
        :param petal_width: The width of the petal.
        :param species: The spicies of the flower.
        """
        super().__init__(sepal_length, sepal_width, petal_length, petal_width)
        self.species = species

    def __repr__(self) -> str:
        if self.classification is None:
            classification = ""
        else:
            classification = f", classification={self.classification}"
        return (
            f"KnownSample("
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
    def from_dict(cls, row: dict[str, str]) -> KnownSample:
        """Create a KnownSample from a dictionary."""
        if row["species"] not in {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}:
            raise InvalidSampleError(f"Invalid species in row: {row!r}")

        try:
            return cls(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
                species=row["species"],
            )
        except ValueError:
            raise InvalidSampleError(f"Invalid sample in row: {row!r}")


class UnknownSample(Sample):
    """A sample of an iris flower with an unknown spicies."""

    def __repr__(self) -> str:
        return super().__repr__().replace("Sample", "UnknownSample")
