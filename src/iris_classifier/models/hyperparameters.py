"""Hyperparameter sets for the model."""
from __future__ import annotations

import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from iris_classifier.distances import Distance
    from iris_classifier.training import TrainingData

from .samples import Sample, TrainingKnownSample


@dataclass
class Hyperparameter:
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    k: int
    algorithm: Distance
    data: weakref.ReferenceType[TrainingData]
    quality: float | None = field(default=None, init=False)

    def test(self) -> None:
        """Test the hyperparameter set."""
        training_data: Optional[TrainingData] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample.sample.sample)
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
        training_data: Optional[TrainingData] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            [
                (
                    self.algorithm.distance(sample, training_sample.sample.sample),
                    training_sample,
                )
                for training_sample in training_data.training
            ],
            key=lambda x: x[0],
        )
        k_nearest_species = (known.sample.species for d, known in distances[: self.k])
        frequencies: Counter[str] = Counter(k_nearest_species)
        best_fit, *_ = frequencies.most_common()
        species, _ = best_fit
        return species
