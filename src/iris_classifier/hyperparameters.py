"""Hyperparameter sets for the model."""

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
