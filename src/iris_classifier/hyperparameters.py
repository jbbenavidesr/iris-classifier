"""Hyperparameter sets for the model."""
from __future__ import annotations

import weakref
from collections import Counter
from typing import Optional

from .samples import Sample, Species, TrainingKnownSample
from .training import TrainingData


class Hyperparameter:
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    def __init__(self, k: int, distance: Distance, training: TrainingData) -> None:
        """Initialize a hyperparameter set.

        :param k: The k value for the model.
        :param training: The training data used to train the model.
        """
        self.k = k
        self.distance = distance
        self.data: weakref.ReferenceType[TrainingData] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Test the hyperparameter set."""
        training_data: Optional[TrainingData] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1

        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> Species:
        """Classify a sample.

        :param sample: The sample to classify.
        :return: The classification of the sample.
        """
        training_data: Optional[TrainingData] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            [
                (self.distance.distance(sample, training_sample), training_sample)
                for training_sample in training_data.training
            ],
            key=lambda x: x[0],
        )
        k_nearest_species = (known.species for d, known in distances[: self.k])
        frequencies: Counter[Species] = Counter(k_nearest_species)
        best_fit, *_ = frequencies.most_common()
        species, _ = best_fit
        return species


class Distance:
    """Definition of a distance computation."""

    def distance(self, a: Sample, b: Sample) -> float:
        """Compute the distance between two samples.

        :param a: The first sample.
        :param b: The second sample.
        :return: The distance between the samples.
        """
        raise NotImplementedError


class MinkowskiDistance(Distance):
    """Minkowski distance."""

    p: int

    def distance(self, a: Sample, b: Sample) -> float:
        return sum(
            [
                abs(a.sepal_length - b.sepal_length) ** self.p,
                abs(a.sepal_width - b.sepal_width) ** self.p,
                abs(a.petal_length - b.petal_length) ** self.p,
                abs(a.petal_width - b.petal_width) ** self.p,
            ]
        ) ** (1 / self.p)


class ED(MinkowskiDistance):
    """Euclidean distance."""

    p = 2


class MD(MinkowskiDistance):
    """Manhattan distance."""

    p = 1


class CD(Distance):
    """Chebyshev distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return max(
            [
                abs(a.sepal_length - b.sepal_length),
                abs(a.sepal_width - b.sepal_width),
                abs(a.petal_length - b.petal_length),
                abs(a.petal_width - b.petal_width),
            ]
        )


class SD(Distance):
    """Sorensen distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return sum(
            [
                abs(a.sepal_length - b.sepal_length),
                abs(a.sepal_width - b.sepal_width),
                abs(a.petal_length - b.petal_length),
                abs(a.petal_width - b.petal_width),
            ]
        ) / sum(
            [
                a.sepal_length + b.sepal_length,
                a.sepal_width + b.sepal_width,
                a.petal_length + b.petal_length,
                a.petal_width + b.petal_width,
            ]
        )
