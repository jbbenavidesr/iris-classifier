import weakref

import pytest

from iris_classifier.distances import EuclidianDistance as ED
from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.training import TrainingData
from iris_classifier.samples import Sample, KnownSample

from .mock_data import training_data as mock_training_data


@pytest.fixture(scope="module")
def training_data() -> TrainingData:
    """Return a training data."""
    training_data = TrainingData("Test training data")
    training_data.load(mock_training_data)
    return training_data


@pytest.fixture(scope="module")
def hyperparameter(training_data: TrainingData) -> Hyperparameter:
    """Return a hyperparameter."""
    distance = ED()
    return Hyperparameter(3, distance, weakref.ref(training_data))


@pytest.fixture(scope="module")
def sample() -> Sample:
    """Return a sample."""
    return Sample(1.0, 2.0, 3.0, 4.0)


@pytest.fixture(scope="module")
def known_sample() -> KnownSample:
    """Return a known sample."""
    return KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
