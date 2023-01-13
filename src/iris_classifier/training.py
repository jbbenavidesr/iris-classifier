"""Simple application for classifying iris flowers.

This application will have 2 users: A botanist, which is the one who will add the
classified data of iris flowers that will be used to train the model. And a user which
wants an unknown sample of iris flower to be classified by the application. The botanist
will select the hyperparameters used by the model.
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from .models import (
        Hyperparameter,
        KnownSample,
        TestingKnownSample,
        TrainingKnownSample,
    )

from .partitions import partition_samples, training_90


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
        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(
        self,
        raw_data_source: Iterable[KnownSample],
        partition_rule: Callable[[int], bool] = training_90,
    ) -> None:
        """Load the raw data source and partition it into training and testing data.

        :return: The number of samples loaded and the number of invalid samples.
        """
        training, testing = partition_samples(raw_data_source, partition_rule)
        self.training = training
        self.testing = testing
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """Test the hyperparameter set.

        :param parameter: The hyperparameter set to test.
        """
        parameter.test(self.testing)
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)
