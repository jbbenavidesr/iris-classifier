"""Hyperparameter sets for the model."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from iris_classifier.distances import DistanceFunc
    from iris_classifier.classifiers import Classifier

from .samples import (
    TrainingList,
    AnySample,
    TestingList,
)


class Hyperparameter(NamedTuple):
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    k: int
    distance_function: DistanceFunc
    training_data: TrainingList
    classifier: Classifier

    def classify(self, unknown: AnySample) -> str:
        """Classify a sample.

        :param sample: The sample to classify.
        :return: The classification of the sample.
        """
        classifier = self.classifier
        return classifier(self.k, self.distance_function, self.training_data, unknown)

    def test(self, testing: TestingList) -> int:

        pass_fail = map(
            lambda t: (1 if t.sample.species == self.classify(t.sample) else 0),
            testing,
        )
        return sum(pass_fail)
