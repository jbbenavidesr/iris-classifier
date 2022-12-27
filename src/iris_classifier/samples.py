from __future__ import annotations

import enum
from typing import cast


class Species(str, enum.Enum):
    SETOSA = "Iris-setosa"
    VERSICOLOR = "Iris-versicolor"
    VIRGINICA = "Iris-virginica"


class Purpose(enum.IntEnum):
    CLASSIFICATION = 0
    TESTING = 1
    TRAINING = 2


class Sample:
    """A sample of an iris flower. Base class used for all types of samples"""

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

    @property
    def classification(self) -> Species | None:
        """Return the classification of the sample."""
        raise NotImplementedError

    @classification.setter
    def classification(self, value: Species | str) -> None:
        """Set the classification of the sample."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) != type(self):
            return False
        other = cast(Sample, other)
        return all(
            [
                self.sepal_length == other.sepal_length,
                self.sepal_width == other.sepal_width,
                self.petal_length == other.petal_length,
                self.petal_width == other.petal_width,
            ]
        )

    @property
    def attr_dict(self) -> dict[str, str]:
        return dict(
            sepal_length=f"{self.sepal_length!r}",
            sepal_width=f"{self.sepal_width!r}",
            petal_length=f"{self.petal_length!r}",
            petal_width=f"{self.petal_width!r}",
        )

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


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

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.species == self.classification

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
        base_attributes = self.attr_dict
        base_attributes["purpose"] = f"{self.purpose.value}"
        base_attributes["species"] = f"{self.species.value}"
        if self.purpose == Purpose.TESTING and self._classification:
            base_attributes["classification"] = f"{self._classification.value}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


class UnknownSample(Sample):
    """A sample of an iris flower with an unknown spicies."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(sepal_length, sepal_width, petal_length, petal_width)
        self._classification: Species | None = None

    @property
    def classification(self) -> Species | None:
        return self._classification

    @classification.setter
    def classification(self, value: Species | str) -> None:
        self._classification = Species(value)

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"
