from __future__ import annotations

import enum
from typing import TypedDict


class Species(str, enum.Enum):
    SETOSA = "Iris-setosa"
    VERSICOLOR = "Iris-versicolor"
    VIRGINICA = "Iris-virginica"


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


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
        species: Species | str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        """Initialize a sample of an iris flower with a known spicies.

        :param sepal_length: The length of the sepal.
        :param sepal_width: The width of the sepal.
        :param petal_length: The length of the petal.
        :param petal_width: The width of the petal.
        :param purpose: The purpose of the sample.
        :param species: The spicies of the flower.
        """
        super().__init__(sepal_length, sepal_width, petal_length, petal_width)
        self.species = Species(species)

    @property
    def attr_dict(self) -> dict[str, str]:
        return {**super().attr_dict, "species": f"{self.species.value}"}


class TrainingKnownSample(KnownSample):
    """A sample of an iris flower with a known spicies used for training the model."""

    pass


class TestingKnownSample(KnownSample):
    """A sample of an iris flower with a known spicies used for testing the model."""

    __test__ = False

    def __init__(
        self,
        species: Species | str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        classification: Species | str | None = None,
    ) -> None:
        super().__init__(species, sepal_length, sepal_width, petal_length, petal_width)
        self.classification = (
            Species(classification) if classification is not None else None
        )

    @property
    def attr_dict(self) -> dict[str, str]:
        attrs = super().attr_dict
        if self.classification is not None:
            attrs["classification"] = f"{self.classification.value}"
        return attrs

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.species == self.classification


class UnknownSample(Sample):
    """A sample of an iris flower with an unknown spicies."""

    pass


class ClassifiedSample(Sample):
    """Created from the sample provided by the user and the result of classification"""

    def __init__(self, classification: Species | str, sample: UnknownSample) -> None:
        super().__init__(
            sample.sepal_length,
            sample.sepal_width,
            sample.petal_length,
            sample.petal_width,
        )
        self.classification = Species(classification)

    @property
    def attr_dict(self) -> dict[str, str]:
        return {**super().attr_dict, "classification": f"{self.classification.value}"}
