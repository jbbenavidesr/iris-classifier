import weakref

import pytest

from iris_classifier.distances import EuclidianDistance as ED
from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.samples import KnownSample, Sample
from iris_classifier.training import TrainingData


@pytest.fixture(scope="module")
def mock_training_data() -> list[KnownSample]:
    """Return a mock training data."""
    return [
        KnownSample(Sample(5.1, 3.5, 1.4, 0.2), "Iris-setosa"),
        KnownSample(Sample(4.9, 3.0, 1.4, 0.2), "Iris-versicolor"),
        KnownSample(Sample(4.7, 3.2, 1.3, 0.2), "Iris-virginica"),
        KnownSample(Sample(4.6, 3.1, 1.5, 0.2), "Iris-setosa"),
        KnownSample(Sample(5.0, 3.6, 1.4, 0.2), "Iris-versicolor"),
        KnownSample(Sample(5.4, 3.9, 1.7, 0.4), "Iris-virginica"),
        KnownSample(Sample(4.6, 3.4, 1.4, 0.3), "Iris-setosa"),
        KnownSample(Sample(5.0, 3.4, 1.5, 0.2), "Iris-versicolor"),
        KnownSample(Sample(4.4, 2.9, 1.4, 0.2), "Iris-virginica"),
        KnownSample(Sample(4.9, 3.1, 1.5, 0.1), "Iris-setosa"),
        KnownSample(Sample(5.4, 3.7, 1.5, 0.2), "Iris-versicolor"),
        KnownSample(Sample(4.8, 3.4, 1.6, 0.2), "Iris-virginica"),
        KnownSample(Sample(4.8, 3.0, 1.4, 0.1), "Iris-setosa"),
        KnownSample(Sample(4.3, 3.0, 1.1, 0.1), "Iris-versicolor"),
        KnownSample(Sample(5.8, 4.0, 1.2, 0.2), "Iris-virginica"),
        KnownSample(Sample(5.7, 4.4, 1.5, 0.4), "Iris-setosa"),
        KnownSample(Sample(5.4, 3.9, 1.3, 0.4), "Iris-versicolor"),
        KnownSample(Sample(5.1, 3.5, 1.4, 0.3), "Iris-virginica"),
        KnownSample(Sample(5.7, 3.8, 1.7, 0.3), "Iris-setosa"),
        KnownSample(Sample(5.1, 3.8, 1.5, 0.3), "Iris-versicolor"),
        KnownSample(Sample(5.4, 3.4, 1.7, 0.2), "Iris-virginica"),
        KnownSample(Sample(5.1, 3.7, 1.5, 0.4), "Iris-setosa"),
        KnownSample(Sample(4.6, 3.6, 1.0, 0.2), "Iris-versicolor"),
        KnownSample(Sample(5.1, 3.3, 1.7, 0.5), "Iris-virginica"),
        KnownSample(Sample(4.8, 3.4, 1.9, 0.2), "Iris-setosa"),
        KnownSample(Sample(5.0, 3.0, 1.6, 0.2), "Iris-versicolor"),
        KnownSample(Sample(5.0, 3.4, 1.6, 0.4), "Iris-virginica"),
        KnownSample(Sample(5.2, 3.5, 1.5, 0.2), "Iris-setosa"),
        KnownSample(Sample(5.2, 3.4, 1.4, 0.2), "Iris-versicolor"),
        KnownSample(Sample(4.7, 3.2, 1.6, 0.2), "Iris-virginica"),
    ]


@pytest.fixture(scope="module")
def training_data(mock_training_data: list[KnownSample]) -> TrainingData:
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
def known_sample(sample: Sample) -> KnownSample:
    """Return a known sample."""
    return KnownSample(sample, "Iris-setosa")
