import random

import pytest

from iris_classifier.distances import EuclidianDistance as ED
from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.training import TrainingData

from .mock_data import training_data as mock_training_data

random.seed(42)


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
    return Hyperparameter(3, distance, training_data)
