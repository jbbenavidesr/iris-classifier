"""Simple application for classifying iris flowers.

This application will have 2 users: A botanist, which is the one who will add the
classified data of iris flowers that will be used to train the model. And a user which
wants an unknown sample of iris flower to be classified by the application. The botanist
will select the hyperparameters used by the model.
"""
from __future__ import annotations

import datetime
import random
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .hyperparameters import Hyperparameter

from .abstracts import SamplePartition
from .samples import KnownSample, Purpose, Sample, Species, SampleDict


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

    def load(
        self, raw_data_source: Iterable[SampleDict], partition: SamplePartition
    ) -> None:
        """Load the raw data source and partition it into training and testing data.

        :return: The number of samples loaded and the number of invalid samples.
        """
        for sample in raw_data_source:
            partition.append(sample)

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

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify a sample.

        :param sample: The sample to classify.
        :param parameter: The hyperparameter set to use.
        :return: The classified sample.
        """
        classification = parameter.classify(sample)
        sample.classification = classification
        return sample


class ShufflingSamplePartition(SamplePartition):
    """Implementation of partition that shuffles the list and cuts it."""

    def __init__(
        self,
        iterable: Iterable[SampleDict] | None = None,
        *,
        training_subset: float = 0.8,
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: int | None = None

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)

    @property
    def training(self) -> list[KnownSample]:
        self.shuffle()
        return [
            KnownSample(**sample, purpose=Purpose.TRAINING)
            for sample in self[: self.split]
        ]

    @property
    def testing(self) -> list[KnownSample]:
        self.shuffle()
        return [
            KnownSample(**sample, purpose=Purpose.TESTING)
            for sample in self[self.split :]
        ]
