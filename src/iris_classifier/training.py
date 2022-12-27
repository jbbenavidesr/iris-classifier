"""Simple application for classifying iris flowers.

This application will have 2 users: A botanist, which is the one who will add the
classified data of iris flowers that will be used to train the model. And a user which
wants an unknown sample of iris flower to be classified by the application. The botanist
will select the hyperparameters used by the model.
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .hyperparameters import Hyperparameter

from .samples import KnownSample, Purpose, Sample, Species


class TrainingData:
    """Set of training  and testing data used to train the model. It has methods to load
    and test the samples."""

    def __init__(self, name: str) -> None:
        """Initialize the training data.

        :param name: The name of the training data.
        """
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[KnownSample] = []
        self.testing: list[KnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_source: Iterable[dict[str, str]]) -> tuple[int, int]:
        """Load the raw data source and partition it into training and testing data.

        :return: The number of samples loaded and the number of invalid samples.
        """
        bad_count = 0
        good_count = 0
        for n, raw_sample in enumerate(raw_data_source):
            purpose = Purpose.TRAINING if n % 10 else Purpose.TESTING
            try:
                sample = KnownSample(
                    sepal_length=float(raw_sample["sepal_length"]),
                    sepal_width=float(raw_sample["sepal_width"]),
                    petal_length=float(raw_sample["petal_length"]),
                    petal_width=float(raw_sample["petal_width"]),
                    species=Species(raw_sample["species"]),
                    purpose=purpose,
                )
                if sample.purpose == Purpose.TRAINING:
                    self.training.append(sample)
                else:
                    self.testing.append(sample)
                good_count += 1
            except ValueError as exc:
                print(f"Invalid sample found in row {n + 1}: {exc}")
                bad_count += 1
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)
        return good_count, bad_count

    def test(self, parameter: Hyperparameter) -> None:
        """Test the hyperparameter set.

        :param parameter: The hyperparameter set to test.
        """
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify a sample.

        :param sample: The sample to classify.
        :param parameter: The hyperparameter set to use.
        :return: The classified sample.
        """
        classification = parameter.classify(sample)
        sample.classification = classification
        return sample
