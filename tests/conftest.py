import pytest
import random

from iris_classifier.hyperparameters import ED, Hyperparameter
from iris_classifier.training import TrainingData, ShufflingSamplePartition

from .mock_data import training_data as mock_training_data

random.seed(42)


@pytest.fixture(scope="module")
def training_data() -> TrainingData:
    """Return a training data."""
    training_data = TrainingData("Test training data")
    training_data.load(
        mock_training_data, ShufflingSamplePartition(training_subset=0.9)
    )
    return training_data


@pytest.fixture(scope="module")
def hyperparameter(training_data: TrainingData) -> Hyperparameter:
    """Return a hyperparameter."""
    distance = ED()
    return Hyperparameter(3, distance, training_data)
