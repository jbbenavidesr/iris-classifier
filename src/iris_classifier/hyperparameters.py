"""Hyperparameter sets for the model."""
from __future__ import annotations

import weakref
from collections import Counter
from typing import Optional

from .distances import Distance
from .samples import Sample, Species, TrainingKnownSample
from .training import TrainingData


class Hyperparameter:
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    def __init__(self, k: int, algorithm: Distance, training: TrainingData) -> None:
        """Initialize a hyperparameter set.

        :param k: The k value for the model.
        :param training: The training data used to train the model.
        """
        self.k = k
        self.algorithm = algorithm
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
                (self.algorithm.distance(sample, training_sample), training_sample)
                for training_sample in training_data.training
            ],
            key=lambda x: x[0],
        )
        k_nearest_species = (known.species for d, known in distances[: self.k])
        frequencies: Counter[Species] = Counter(k_nearest_species)
        best_fit, *_ = frequencies.most_common()
        species, _ = best_fit
        return species
