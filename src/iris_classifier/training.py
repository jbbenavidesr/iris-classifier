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

from .partitions import CountingDealingPartition
from .samples import (
    ClassifiedSample,
    SampleDict,
    TestingKnownSample,
    TrainingKnownSample,
    UnknownSample,
)


class TrainingData:
    """Set of training  and testing data used to train the model. It has methods to load
    and test the samples."""

    partition_class = CountingDealingPartition

    def __init__(self, name: str) -> None:
        """Initialize the training data.

        :param name: The name of the training data.
        """
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_source: Iterable[SampleDict]) -> None:
        """Load the raw data source and partition it into training and testing data.

        :return: The number of samples loaded and the number of invalid samples.
        """
        partition_class = self.partition_class
        partition = partition_class(raw_data_source, training_subset=(9, 10))
        self.training = partition.training
        self.testing = partition.testing
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """Test the hyperparameter set.

        :param parameter: The hyperparameter set to test.
        """
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(
        self, parameter: Hyperparameter, sample: UnknownSample
    ) -> ClassifiedSample:
        """Classify a sample.

        :param sample: The sample to classify.
        :param parameter: The hyperparameter set to use.
        :return: The classified sample.
        """
        classification = parameter.classify(sample)
        return ClassifiedSample(classification, sample)
