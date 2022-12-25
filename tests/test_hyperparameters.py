from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from iris_classifier.training import TrainingData

from iris_classifier.hyperparameters import CD, ED, MD, SD, Distance, Hyperparameter
from iris_classifier.samples import Sample


def test_hyperparameter_init(training_data: TrainingData) -> None:
    """Test the initialization of a hyperparameter."""
    distance = ED()
    hyperparameter = Hyperparameter(3, distance, training_data)
    assert hyperparameter.k == 3
    assert hyperparameter.distance == distance
    assert hyperparameter.data() == training_data


def test_hyperparameter_classify(training_data: TrainingData) -> None:
    distance = ED()
    hyperparameter = Hyperparameter(3, distance, training_data)
    assert hyperparameter.classify(training_data.training[0]) == "Iris-setosa"


def test_hyperparameter_test(training_data: TrainingData) -> None:
    """Test the testing of a hyperparameter."""
    iris_setosa_in_testing = 0
    for sample in training_data.testing:
        if sample.spicies == "Iris-setosa":
            iris_setosa_in_testing += 1

    quality = iris_setosa_in_testing / len(training_data.testing)
    distance = ED()
    hyperparameter = Hyperparameter(3, distance, training_data)
    hyperparameter.test()
    assert hyperparameter.quality == quality


def test_distance_class_throws_not_implemented_error() -> None:
    """Test that the Distance class throws a NotImplementedError."""
    distance = Distance()
    sample_a = Sample(1, 2, 3, 4, "Iris-setosa")
    sample_b = Sample(5, 6, 7, 8, "Iris-setosa")
    with pytest.raises(NotImplementedError):
        distance.distance(sample_a, sample_b)


def test_euclidean_distance() -> None:
    """Test the euclidean distance."""
    ed = ED()
    sample_a = Sample(1, 2, 3, 4, "Iris-setosa")
    sample_b = Sample(5, 6, 7, 8, "Iris-setosa")
    assert ed.distance(sample_a, sample_b) == 8


def test_manhattan_distance() -> None:
    """Test the manhattan distance."""
    md = MD()
    sample_a = Sample(1, 2, 3, 4, "Iris-setosa")
    sample_b = Sample(5, 6, 7, 8, "Iris-setosa")
    assert md.distance(sample_a, sample_b) == 16


def test_chebyshev_distance() -> None:
    """Test the chebyshev distance."""
    cd = CD()
    sample_a = Sample(1, 2, 3, 4, "Iris-setosa")
    sample_b = Sample(5, 6, 7, 8, "Iris-setosa")
    assert cd.distance(sample_a, sample_b) == 4


def test_sorensen_distance() -> None:
    """Test the sorensen distance."""
    sd = SD()
    sample_a = Sample(1, 2, 3, 4, "Iris-setosa")
    sample_b = Sample(5, 6, 7, 8, "Iris-setosa")
    assert sd.distance(sample_a, sample_b) == 0.4444444444444444
