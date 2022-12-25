"""Hyperparameter sets for the model."""

import math
import weakref
from typing import Optional

from .samples import Sample
from .training import TrainingData


class Hyperparameter:
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    def __init__(self, k: int, training: TrainingData) -> None:
        """Initialize a hyperparameter set.

        :param k: The k value for the model.
        :param training: The training data used to train the model.
        """
        self.k = k
        self.data: weakref.ReferenceType[TrainingData] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Test the hyperparameter set."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classify(self.classify(sample))
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1

        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """Classify a sample.

        :param sample: The sample to classify.
        :return: The classification of the sample.
        """
        # For testing
        return "Iris-setosa"


class Distance:
    """Definition of a distance computation."""

    def distance(self, a: Sample, b: Sample) -> float:
        """Compute the distance between two samples.

        :param a: The first sample.
        :param b: The second sample.
        :return: The distance between the samples.
        """
        raise NotImplementedError


class ED(Distance):
    """Euclidean distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return math.hypot(
            a.sepal_length - b.sepal_length,
            a.sepal_width - b.sepal_width,
            a.petal_length - b.petal_length,
            a.petal_width - b.petal_width,
        )


class MD(Distance):
    """Manhattan distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return sum(
            [
                abs(a.sepal_length - b.sepal_length),
                abs(a.sepal_width - b.sepal_width),
                abs(a.petal_length - b.petal_length),
                abs(a.petal_width - b.petal_width),
            ]
        )


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
